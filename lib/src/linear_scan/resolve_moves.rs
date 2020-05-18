use super::{next_use, IntId, Location, RegUses, VirtualInterval};
use crate::{
    data_structures::{BlockIx, InstIx, InstPoint, Point},
    inst_stream::{InstToInsert, InstToInsertAndPoint},
    sparse_set::SparseSet,
    Function, RealReg, Reg, SpillSlot, TypedIxVec, VirtualReg, Writable,
};

use log::{debug, info, trace};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use smallvec::SmallVec;
use std::fmt;

fn resolve_moves_in_block(
    intervals: &Vec<VirtualInterval>,
    reg_uses: &RegUses,
    block_starts: &HashSet<InstIx>,
    block_ends: &HashSet<InstIx>,
    scratches_by_rc: &[Option<RealReg>],
    spill_slot: &mut u32,
    ordered_moves: &mut Vec<MoveOp>,
    memory_moves: &mut HashMap<InstPoint, SmallVec<[InstToInsert; 8]>>,
    tmp_stack: &mut Vec<MoveOp>,
) {
    let mut reloads_at_inst = HashMap::default();
    let mut spills_at_inst = HashMap::default();

    for interval in intervals {
        let parent_id = match interval.parent {
            Some(pid) => pid,
            None => {
                // In unreachable code, it's possible that a given interval has no
                // parents and is assigned to a stack location for its whole lifetime.
                //
                // In reachable code, the analysis only create intervals for virtual
                // registers with at least one register use, so a parentless interval (=
                // hasn't ever been split) can't live in a stack slot.
                debug_assert!(
                    interval.location.spill().is_none()
                        || (next_use(interval, InstPoint::min_value(), reg_uses,).is_none())
                );
                continue;
            }
        };

        let parent = &intervals[parent_id.0];

        // If this is a move between blocks, handle it as such.
        if parent.end.pt() == Point::Def
            && interval.start.pt() == Point::Use
            && block_ends.contains(&parent.end.iix())
            && block_starts.contains(&interval.start.iix())
        {
            continue;
        }

        let child_start = interval.start;
        let vreg = interval.vreg;

        match interval.location {
            Location::None => panic!("interval has no location after regalloc!"),

            Location::Reg(rreg) => {
                // Reconnect with the parent location, by adding a move if needed.
                if let Some(next_use) = next_use(interval, child_start, reg_uses) {
                    // No need to reload before a new definition.
                    if next_use.pt() == Point::Def {
                        continue;
                    }
                };

                let mut at_inst = child_start;
                match at_inst.pt() {
                    Point::Use => {
                        at_inst.set_pt(Point::Reload);
                    }
                    Point::Def => {
                        at_inst.set_pt(Point::Spill);
                    }
                    _ => unreachable!(),
                }

                let entry = reloads_at_inst.entry(at_inst).or_insert(Vec::new());

                match parent.location {
                    Location::None => unreachable!(),

                    Location::Reg(from_rreg) => {
                        if from_rreg != rreg {
                            debug!(
                                "inblock fixup: {:?} move {:?} -> {:?} at {:?}",
                                interval.id, from_rreg, rreg, at_inst
                            );
                            entry.push(MoveOp::new_move(from_rreg, rreg, vreg));
                        }
                    }

                    Location::Stack(spill) => {
                        debug!(
                            "inblock fixup: {:?} reload {:?} -> {:?} at {:?}",
                            interval.id, spill, rreg, at_inst
                        );
                        entry.push(MoveOp::new_reload(spill, rreg, vreg));
                    }
                }
            }

            Location::Stack(spill) => {
                // This interval has been spilled (i.e. split). Spill after the last def or before
                // the last use.
                let mut at_inst = parent.end;
                at_inst.set_pt(if at_inst.pt() == Point::Use {
                    Point::Reload
                } else {
                    debug_assert!(at_inst.pt() == Point::Def);
                    Point::Spill
                });

                match parent.location {
                    Location::None => unreachable!(),

                    Location::Reg(rreg) => {
                        debug!(
                            "inblock fixup: {:?} spill {:?} -> {:?} at {:?}",
                            interval.id, rreg, spill, at_inst
                        );
                        spills_at_inst.entry(at_inst).or_insert(Vec::new()).push(
                            InstToInsert::Spill {
                                to_slot: spill,
                                from_reg: rreg,
                                for_vreg: vreg,
                            },
                        );
                    }

                    Location::Stack(parent_spill) => {
                        debug_assert_eq!(parent_spill, spill);
                    }
                }
            }
        }
    }

    // Flush the memory moves caused by in-block fixups. Conceptually, the spills
    // must happen after the right locations have been set, that is, after the
    // reloads. Reloads may include several moves that must happen in parallel
    // (e.g. if two real regs must be swapped), so process them first. Once all
    // the parallel assignments have been done, push forward all the spills.
    for (at_inst, mut pending_moves) in reloads_at_inst {
        schedule_moves(&mut pending_moves, ordered_moves, tmp_stack);
        let move_insts = emit_moves(&ordered_moves, spill_slot, scratches_by_rc);
        memory_moves.insert(at_inst, move_insts);
    }

    for (at_inst, spills) in spills_at_inst {
        memory_moves
            .entry(at_inst)
            .or_insert(SmallVec::new())
            .extend(spills.into_iter());
    }
}

fn resolve_moves_accross_blocks<F: Function>(
    func: &F,
    liveins: &TypedIxVec<BlockIx, SparseSet<Reg>>,
    liveouts: &TypedIxVec<BlockIx, SparseSet<Reg>>,
    sorted_intervals: &Vec<VirtualInterval>,
    intervals: &Vec<VirtualInterval>,
    scratches_by_rc: &[Option<RealReg>],
    spill_slot: &mut u32,
    ordered_moves: &mut Vec<MoveOp>,
    memory_moves: &mut HashMap<InstPoint, SmallVec<[InstToInsert; 8]>>,
    tmp_stack: &mut Vec<MoveOp>,
) {
    let mut parallel_move_map = HashMap::default();
    enum BlockPos {
        Start,
        End,
    }

    // Figure the sequence of parallel moves to insert at block boundaries:
    // - for each block
    //  - for each liveout vreg in this block
    //    - for each successor of this block
    //      - if the locations allocated in the block and its successor don't
    //      match, insert a pending move from one location to the other.
    //
    // Once that's done:
    // - resolve cycles in the pending moves
    // - generate real moves from the pending moves.
    let mut seen_successors = HashSet::default();
    for block in func.blocks() {
        let successors = func.block_succs(block);

        // Where to insert the fixup move, if needed? If there's more than one
        // successor to the current block, inserting in the current block will
        // impact all the successors.
        //
        // We assume critical edges have been split, so
        // if the current block has more than one successor, then its successors
        // have at most one predecessor.
        let cur_has_one_succ = successors.len() == 1;

        for &reg in liveouts[block].iter() {
            let vreg = if let Some(vreg) = reg.as_virtual_reg() {
                vreg
            } else {
                continue;
            };

            seen_successors.clear();

            let mut maybe_cur_id: Option<IntId> = None;

            let last_inst = func.block_insns(block).last();
            let cur_last_inst = InstPoint::new_def(last_inst);

            for &succ in successors.iter() {
                if !liveins[succ].contains(reg) {
                    // This variable isn't live in this block.
                    continue;
                }
                if !seen_successors.insert(succ) {
                    continue;
                }

                // Find the interval for this (vreg, inst) pair.
                let succ_first_inst = InstPoint::new_use(func.block_insns(succ).first());

                if let Some(cur_id) = maybe_cur_id {
                    if intervals[cur_id.0].covers(succ_first_inst) {
                        continue;
                    }
                }

                let succ_id = find_enclosing_interval(vreg, succ_first_inst, sorted_intervals)
                    .expect("variable should have been live in successor");

                // Fast-path: if the same intervals covers the last instruction, we're within a
                // single live interval and don't need a move.
                if intervals[succ_id.0].covers(cur_last_inst) {
                    continue;
                }

                let cur_id = match maybe_cur_id {
                    None => {
                        let id = find_enclosing_interval(vreg, cur_last_inst, &sorted_intervals)
                            .unwrap_or_else(|| {
                                panic!(
                                    "no interval for given {:?}:{:?} pair in current {:?}",
                                    vreg, cur_last_inst, block
                                )
                            });
                        maybe_cur_id = Some(id);
                        id
                    }
                    Some(id) => id,
                };

                if intervals[cur_id.0].ancestor(&intervals)
                    != intervals[succ_id.0].ancestor(&intervals)
                {
                    continue;
                }

                let (at_inst, block_pos) = if cur_has_one_succ {
                    let mut pos = cur_last_inst;
                    // Before the control flow instruction.
                    pos.set_pt(Point::Reload);
                    (pos, BlockPos::End)
                } else {
                    let mut pos = succ_first_inst;
                    pos.set_pt(Point::Reload);
                    (pos, BlockPos::Start)
                };

                let pending_moves = parallel_move_map
                    .entry(at_inst)
                    .or_insert((Vec::new(), block_pos));

                match (intervals[cur_id.0].location, intervals[succ_id.0].location) {
                    (Location::Reg(cur_rreg), Location::Reg(succ_rreg)) => {
                        if cur_rreg == succ_rreg {
                            continue;
                        }
                        debug!(
                          "boundary fixup: move {:?} -> {:?} at {:?} for {:?} between {:?} and {:?}",
                          cur_rreg,
                          succ_rreg,
                          at_inst,
                          vreg,
                          block,
                          succ
                        );
                        pending_moves
                            .0
                            .push(MoveOp::new_move(cur_rreg, succ_rreg, vreg));
                    }

                    (Location::Reg(cur_rreg), Location::Stack(spillslot)) => {
                        debug!(
                          "boundary fixup: spill {:?} -> {:?} at {:?} for {:?} between {:?} and {:?}",
                          cur_rreg,
                          spillslot,
                          at_inst,
                          vreg,
                          block,
                          succ
                        );
                        pending_moves
                            .0
                            .push(MoveOp::new_spill(cur_rreg, spillslot, vreg));
                    }

                    (Location::Stack(spillslot), Location::Reg(rreg)) => {
                        debug!(
                          "boundary fixup: reload {:?} -> {:?} at {:?} for {:?} between {:?} and {:?}",
                          spillslot,
                          rreg,
                          at_inst,
                          vreg,
                          block,
                          succ
                        );
                        pending_moves
                            .0
                            .push(MoveOp::new_reload(spillslot, rreg, vreg));
                    }

                    (Location::Stack(left_spill_slot), Location::Stack(right_spill_slot)) => {
                        // Stack to stack should not happen here, since two ranges for the
                        // same vreg can't be intersecting, so the same stack slot ought to
                        // be reused in this case.
                        debug_assert_eq!(
                          left_spill_slot, right_spill_slot,
                          "Moves from stack to stack only happen on the same vreg, thus the same stack slot"
                        );
                        continue;
                    }

                    (_, _) => {
                        panic!("register or stack slots must have been allocated.");
                    }
                };
            }
        }

        // Flush the memory moves caused by block fixups for this block.
        for (at_inst, parallel_moves) in parallel_move_map.iter_mut() {
            schedule_moves(&mut parallel_moves.0, ordered_moves, tmp_stack);
            let mut move_insts = emit_moves(&ordered_moves, spill_slot, scratches_by_rc);

            // If at_inst pointed to a block start, then insert block fixups *before* inblock
            // fixups; otherwise it pointed to a block end, then insert block fixups *after*
            // inblock fixups.
            let entry = memory_moves.entry(*at_inst).or_insert(SmallVec::new());
            match parallel_moves.1 {
                BlockPos::Start => {
                    move_insts.extend(entry.iter().cloned());
                    *entry = move_insts;
                }
                BlockPos::End => {
                    entry.extend(move_insts);
                }
            }
        }

        parallel_move_map.clear();
    }

    debug!("");
}

#[inline(never)]
pub(crate) fn run<F: Function>(
    func: &F,
    reg_uses: &RegUses,
    intervals: &Vec<VirtualInterval>,
    sorted_intervals: &Vec<VirtualInterval>,
    liveins: &TypedIxVec<BlockIx, SparseSet<Reg>>,
    liveouts: &TypedIxVec<BlockIx, SparseSet<Reg>>,
    spill_slot: &mut u32,
    scratches_by_rc: &[Option<RealReg>],
) -> Vec<InstToInsertAndPoint> {
    info!("resolve_moves");

    let mut block_ends = HashSet::default();
    let mut block_starts = HashSet::default();
    for bix in func.blocks() {
        let insts = func.block_insns(bix);
        block_ends.insert(insts.last());
        block_starts.insert(insts.first());
    }

    // A global hash-map of all the moves. Note that it will eventually contain both in-block
    // moves, and moves between blocks, which may be reordered at the end.
    let mut memory_moves = HashMap::default();

    let mut tmp_stack = Vec::new();
    let mut ordered_moves = Vec::new();
    resolve_moves_in_block(
        intervals,
        reg_uses,
        &block_starts,
        &block_ends,
        scratches_by_rc,
        spill_slot,
        &mut ordered_moves,
        &mut memory_moves,
        &mut tmp_stack,
    );

    resolve_moves_accross_blocks(
        func,
        liveins,
        liveouts,
        sorted_intervals,
        intervals,
        scratches_by_rc,
        spill_slot,
        &mut ordered_moves,
        &mut memory_moves,
        &mut tmp_stack,
    );

    let mut insts_and_points = Vec::new();
    for (at, insts) in memory_moves {
        for inst in insts {
            insts_and_points.push(InstToInsertAndPoint::new(inst, at));
        }
    }

    insts_and_points
}

#[inline(never)]
fn find_enclosing_interval(
    vreg: VirtualReg,
    inst: InstPoint,
    virtual_intervals: &Vec<VirtualInterval>,
) -> Option<IntId> {
    // The list of virtual intervals is sorted by vreg; find one interval for this vreg.
    let index = virtual_intervals
        .binary_search_by_key(&vreg, |int| int.vreg)
        .expect("should find at least one virtual interval for this vreg");

    // Rewind back to the first interval for this vreg, since there might be several ones.
    let mut index = index;
    while index > 0 && virtual_intervals[index - 1].vreg == vreg {
        index -= 1;
    }

    // Now iterates on all the intervals for this virtual register, until one works.
    let mut int = &virtual_intervals[index];
    loop {
        if int.start <= inst && inst <= int.end {
            return Some(int.id);
        }
        index += 1;
        if index == virtual_intervals.len() {
            return None;
        }
        int = &virtual_intervals[index];
        if int.vreg != vreg {
            return None;
        }
    }
}

#[derive(PartialEq, Debug)]
enum MoveOperand {
    Reg(RealReg),
    Stack(SpillSlot),
}

impl MoveOperand {
    fn aliases(&self, other: &Self) -> bool {
        self == other
    }
}

struct MoveOp {
    from: MoveOperand,
    to: MoveOperand,
    vreg: VirtualReg,
    cycle_begin: Option<usize>,
    cycle_end: Option<usize>,
}

impl fmt::Debug for MoveOp {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}: {:?} -> {:?}", self.vreg, self.from, self.to)?;
        if let Some(ref begin) = self.cycle_begin {
            write!(fmt, ", start of cycle #{}", begin)?;
        }
        if let Some(ref end) = self.cycle_end {
            write!(fmt, ", end of cycle #{}", end)?;
        }
        Ok(())
    }
}

impl MoveOp {
    fn new_move(from: RealReg, to: RealReg, vreg: VirtualReg) -> Self {
        Self {
            from: MoveOperand::Reg(from),
            to: MoveOperand::Reg(to),
            vreg,
            cycle_begin: None,
            cycle_end: None,
        }
    }

    fn new_spill(from: RealReg, to: SpillSlot, vreg: VirtualReg) -> Self {
        Self {
            from: MoveOperand::Reg(from),
            to: MoveOperand::Stack(to),
            vreg,
            cycle_begin: None,
            cycle_end: None,
        }
    }

    fn new_reload(from: SpillSlot, to: RealReg, vreg: VirtualReg) -> Self {
        Self {
            from: MoveOperand::Stack(from),
            to: MoveOperand::Reg(to),
            vreg,
            cycle_begin: None,
            cycle_end: None,
        }
    }

    fn gen_inst(&self) -> InstToInsert {
        match self.from {
            MoveOperand::Reg(from) => match self.to {
                MoveOperand::Reg(to) => InstToInsert::Move {
                    to_reg: Writable::from_reg(to),
                    from_reg: from,
                    for_vreg: self.vreg,
                },
                MoveOperand::Stack(to) => InstToInsert::Spill {
                    to_slot: to,
                    from_reg: from,
                    for_vreg: self.vreg,
                },
            },
            MoveOperand::Stack(from) => match self.to {
                MoveOperand::Reg(to) => InstToInsert::Reload {
                    to_reg: Writable::from_reg(to),
                    from_slot: from,
                    for_vreg: self.vreg,
                },
                MoveOperand::Stack(_to) => unreachable!("stack to stack move"),
            },
        }
    }
}

fn find_blocking_move<'a>(
    pending: &'a mut Vec<MoveOp>,
    last: &MoveOp,
) -> Option<(usize, &'a mut MoveOp)> {
    for (i, other) in pending.iter_mut().enumerate() {
        if other.from.aliases(&last.to) {
            return Some((i, other));
        }
    }
    None
}

fn find_cycled_move<'a>(
    stack: &'a mut Vec<MoveOp>,
    from: &mut usize,
    last: &MoveOp,
) -> Option<&'a mut MoveOp> {
    for i in *from..stack.len() {
        *from += 1;
        let other = &stack[i];
        if other.from.aliases(&last.to) {
            return Some(&mut stack[i]);
        }
    }
    None
}

/// Given a pending list of moves, returns a list of moves ordered in a correct
/// way, i.e., no move clobbers another one.
#[inline(never)]
fn schedule_moves(
    pending: &mut Vec<MoveOp>,
    ordered_moves: &mut Vec<MoveOp>,
    stack: &mut Vec<MoveOp>,
) {
    ordered_moves.clear();

    let mut num_cycles = 0;
    let mut cur_cycles = 0;

    trace!("pending moves: {:#?}", pending);

    while let Some(pm) = pending.pop() {
        trace!("handling pending move {:?}", pm);
        debug_assert!(
            pm.from != pm.to,
            "spurious moves should not have been inserted"
        );

        stack.clear();
        stack.push(pm);

        while !stack.is_empty() {
            let blocking_pair = find_blocking_move(pending, stack.last().unwrap());

            if let Some((blocking_idx, blocking)) = blocking_pair {
                trace!("found blocker: {:?}", blocking);
                let mut stack_cur = 0;

                let has_cycles =
                    if let Some(mut cycled) = find_cycled_move(stack, &mut stack_cur, blocking) {
                        trace!("found cycle: {:?}", cycled);
                        debug_assert!(cycled.cycle_end.is_none());
                        cycled.cycle_end = Some(cur_cycles);
                        true
                    } else {
                        false
                    };

                if has_cycles {
                    loop {
                        match find_cycled_move(stack, &mut stack_cur, blocking) {
                            Some(ref mut cycled) => {
                                trace!("found more cycles ending on blocker: {:?}", cycled);
                                debug_assert!(cycled.cycle_end.is_none());
                                cycled.cycle_end = Some(cur_cycles);
                            }
                            None => break,
                        }
                    }

                    debug_assert!(blocking.cycle_begin.is_none());
                    blocking.cycle_begin = Some(cur_cycles);
                    cur_cycles += 1;
                }

                let blocking = pending.remove(blocking_idx);
                stack.push(blocking);
            } else {
                // There's no blocking move! We can push this in the ordered list of
                // moves.
                // TODO IonMonkey has more optimizations for this case.
                let last = stack.pop().unwrap();
                ordered_moves.push(last);
            }
        }

        if num_cycles < cur_cycles {
            num_cycles = cur_cycles;
        }
        cur_cycles = 0;
    }
}

#[inline(never)]
fn emit_moves(
    ordered_moves: &Vec<MoveOp>,
    num_spill_slots: &mut u32,
    scratches_by_rc: &[Option<RealReg>],
) -> SmallVec<[InstToInsert; 8]> {
    let mut spill_slot = None;
    let mut in_cycle = false;

    let mut move_insts = SmallVec::new();

    trace!("emit_moves");

    for mov in ordered_moves {
        if let Some(_) = &mov.cycle_end {
            debug_assert!(in_cycle);

            // There is some pattern:
            //   (A -> B)
            //   (B -> A)
            // This case handles (B -> A), which we reach last. We emit a move from
            // the saved value of B, to A.
            match mov.to {
                MoveOperand::Reg(dst_reg) => {
                    let inst = InstToInsert::Reload {
                        to_reg: Writable::from_reg(dst_reg),
                        from_slot: spill_slot.expect("should have a cycle spill slot"),
                        for_vreg: mov.vreg,
                    };
                    move_insts.push(inst);
                    trace!(
                        "finishing cycle: {:?} -> {:?}",
                        spill_slot.unwrap(),
                        dst_reg
                    );
                }
                MoveOperand::Stack(dst_spill) => {
                    let scratch = scratches_by_rc[mov.vreg.get_class() as usize]
                        .expect("missing scratch reg");
                    let inst = InstToInsert::Reload {
                        to_reg: Writable::from_reg(scratch),
                        from_slot: spill_slot.expect("should have a cycle spill slot"),
                        for_vreg: mov.vreg,
                    };
                    move_insts.push(inst);
                    let inst = InstToInsert::Spill {
                        to_slot: dst_spill,
                        from_reg: scratch,
                        for_vreg: mov.vreg,
                    };
                    move_insts.push(inst);
                    trace!(
                        "finishing cycle: {:?} -> {:?} -> {:?}",
                        spill_slot.unwrap(),
                        scratch,
                        dst_spill
                    );
                }
            };

            in_cycle = false;
            continue;
        }

        if let Some(_) = &mov.cycle_begin {
            debug_assert!(!in_cycle);

            // There is some pattern:
            //   (A -> B)
            //   (B -> A)
            // This case handles (A -> B), which we reach first. We save B, then allow
            // the original move to continue.
            match spill_slot {
                Some(_) => {}
                None => {
                    spill_slot = Some(SpillSlot::new(*num_spill_slots));
                    *num_spill_slots += 1;
                }
            }

            match mov.to {
                MoveOperand::Reg(src_reg) => {
                    let inst = InstToInsert::Spill {
                        to_slot: spill_slot.unwrap(),
                        from_reg: src_reg,
                        for_vreg: mov.vreg,
                    };
                    move_insts.push(inst);
                    trace!("starting cycle: {:?} -> {:?}", src_reg, spill_slot.unwrap());
                }
                MoveOperand::Stack(src_spill) => {
                    let scratch = scratches_by_rc[mov.vreg.get_class() as usize]
                        .expect("missing scratch reg");
                    let inst = InstToInsert::Reload {
                        to_reg: Writable::from_reg(scratch),
                        from_slot: src_spill,
                        for_vreg: mov.vreg,
                    };
                    move_insts.push(inst);
                    let inst = InstToInsert::Spill {
                        to_slot: spill_slot.expect("should have a cycle spill slot"),
                        from_reg: scratch,
                        for_vreg: mov.vreg,
                    };
                    move_insts.push(inst);
                    trace!(
                        "starting cycle: {:?} -> {:?} -> {:?}",
                        src_spill,
                        scratch,
                        spill_slot.unwrap()
                    );
                }
            };

            in_cycle = true;
        }

        // A normal move which is not part of a cycle.
        move_insts.push(mov.gen_inst());
        trace!("moving {:?} -> {:?}", mov.from, mov.to);
    }

    move_insts
}
