/* -*- Mode: Rust; tab-width: 8; indent-tabs-mode: nil; rust-indent-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
*/
//! Implementation of the linear scan allocator algorithm.

use std::collections::{HashMap, HashSet};
use std::fmt;

use log::debug;

use crate::analysis::run_analysis;
use crate::data_structures::{
  i_reload, i_spill, mkBlockIx, mkInstIx, mkInstPoint, mkRangeFrag,
  mkRangeFragIx, mkRealReg, mkSpillSlot, mkVirtualRangeIx, Block, BlockIx,
  Func, Inst, InstIx, InstPoint, InstPoint_Def, InstPoint_Reload,
  InstPoint_Spill, InstPoint_Use, Map, Point, RangeFrag, RangeFragIx,
  RangeFragKind, RealRange, RealRangeIx, RealReg, RealRegUniverse, Reg,
  RegClass, Set, SortedRangeFragIxs, SpillSlot, TypedIxVec, VirtualRange,
  VirtualRangeIx, VirtualReg, NUM_REG_CLASSES,
};

// Local renamings.
type Fragments = TypedIxVec<RangeFragIx, RangeFrag>;
type VirtualRanges = TypedIxVec<VirtualRangeIx, VirtualRange>;

#[derive(Clone, Copy, PartialEq, Eq)]
struct LiveId(usize);

enum LiveIntervalKind<'a> {
  Fixed(&'a mut RealRange),
  Virtual(&'a mut VirtualRange),
}

impl<'a> fmt::Debug for LiveIntervalKind<'a> {
  fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
    match self {
      LiveIntervalKind::Fixed(range) => write!(fmt, "Fixed({:?})", range),
      LiveIntervalKind::Virtual(range) => write!(fmt, "Virtual({:?})", range),
    }
  }
}

struct LiveInterval<'a> {
  id: LiveId,
  kind: LiveIntervalKind<'a>,
  cur_frag_index: usize,
}

impl<'a> fmt::Debug for LiveInterval<'a> {
  fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
    write!(fmt, "{:?} (frag={})", self.kind, self.cur_frag_index)
  }
}

impl<'a> LiveInterval<'a> {
  fn from_real(id: LiveId, range: &'a mut RealRange) -> Self {
    Self { id, kind: LiveIntervalKind::Fixed(range), cur_frag_index: 0 }
  }
  fn from_virtual(id: LiveId, range: &'a mut VirtualRange) -> Self {
    Self { id, kind: LiveIntervalKind::Virtual(range), cur_frag_index: 0 }
  }

  fn reg_class(&self) -> RegClass {
    match &self.kind {
      LiveIntervalKind::Fixed(r) => r.rreg.get_class(),
      LiveIntervalKind::Virtual(r) => r.vreg.get_class(),
    }
  }
  fn fragments(&self) -> &SortedRangeFragIxs {
    match &self.kind {
      LiveIntervalKind::Fixed(r) => &r.sortedFrags,
      LiveIntervalKind::Virtual(r) => &r.sortedFrags,
    }
  }
  fn allocated_register(&self) -> Option<RealReg> {
    match &self.kind {
      LiveIntervalKind::Fixed(r) => Some(r.rreg),
      LiveIntervalKind::Virtual(r) => r.rreg,
    }
  }

  fn is_fixed(&self) -> bool {
    match &self.kind {
      LiveIntervalKind::Fixed(_) => true,
      _ => false,
    }
  }
  fn fixed_reg(&self) -> Option<RealReg> {
    if self.is_fixed() {
      self.allocated_register()
    } else {
      None
    }
  }

  fn cur_fragment<'frag>(
    &self, fragments: &'frag Fragments,
  ) -> &'frag RangeFrag {
    &fragments[self.fragments().fragIxs[self.cur_frag_index]]
  }
  fn start_point(&self, fragments: &Fragments) -> InstPoint {
    self.cur_fragment(fragments).first
  }
  fn end_point(&self, fragments: &Fragments) -> InstPoint {
    self.cur_fragment(fragments).last
  }
  fn last_end_point(&self, fragments: &Fragments) -> InstPoint {
    fragments[*self.fragments().fragIxs.last().unwrap()].last
  }

  fn cur_frag_covers(&self, pos: InstPoint, fragments: &Fragments) -> bool {
    let cur_frag = self.cur_fragment(fragments);
    cur_frag.first >= pos && pos <= cur_frag.last
  }
  fn has_more_frags(&self) -> bool {
    self.cur_frag_index + 1 < self.fragments().len()
  }

  // Mutators.
  fn move_to_next_frag(&mut self) {
    debug_assert!(self.has_more_frags());
    self.cur_frag_index += 1;
  }

  fn set_reg(&mut self, reg: RealReg) {
    debug_assert!(self.allocated_register().is_none());
    match &mut self.kind {
      LiveIntervalKind::Fixed(_) => unreachable!(),
      LiveIntervalKind::Virtual(ref mut r) => r.rreg = Some(reg),
    }
  }
}

fn update_state<'a>(
  start_point: InstPoint, mut state: State<'a>, fragments: &Fragments,
) -> State<'a> {
  let mut regs = state.regs;

  let mut next_active = Vec::new();
  let mut next_inactive = Vec::new();

  // Update active intervals:
  // - either their current end point is after the start_point,
  //  - and they have other fragments: they become inactive.
  //  - and they don't: they become handled.
  // - or they're still active.
  for &active_int_id in &state.active {
    let active_int = &mut state.intervals[active_int_id];
    if active_int.cur_frag_covers(start_point, &fragments) {
      // Remains active.
      next_active.push(active_int_id);
    } else {
      if active_int.has_more_frags() {
        active_int.move_to_next_frag();

        // XXX do we need to free on active->inactive?
        next_inactive.push(active_int_id);

        // XXX is this needed? bnjbvr continue here
        // Remove from unhandled, and add it back to the next fragment's start
        // position.
        let index =
          state.unhandled.iter().position(|&id| id == active_int_id).unwrap();
        let id = state.unhandled.remove(index);
        insert_unhandled(
          &mut state.unhandled,
          &state.intervals,
          &state.intervals[id],
          fragments,
        );
      } else {
        // Free the register, if it was allocated one.
        if let Some(reg) = active_int.allocated_register() {
          regs[reg.get_class() as usize].free(&reg);
        }

        // TODO move to handled?
      }
    }
  }

  // Update inactive intervals:
  // - either their start point is after the current position, so they remain inactive.
  // - or they remain inactive.
  for &inactive_int_id in &state.inactive {
    let inactive_int = &mut state.intervals[inactive_int_id];
    debug_assert!(inactive_int.has_more_frags());
    if inactive_int.cur_frag_covers(start_point, &fragments) {
      next_active.push(inactive_int_id);
    } else {
      next_inactive.push(inactive_int_id);
    }
  }

  state.active = next_active;
  state.inactive = next_inactive;
  state.regs = regs;

  state
}

fn spill() {
  unimplemented!("spill");
}

#[derive(Clone)]
struct Registers {
  offset: usize,
  // bool: available = true
  regs: Vec<(RealReg, bool)>,
}

impl std::ops::Index<RealReg> for Registers {
  type Output = (RealReg, bool);
  fn index(&self, rreg: RealReg) -> &Self::Output {
    &self.regs[rreg.get_index() - self.offset]
  }
}

impl std::ops::IndexMut<RealReg> for Registers {
  fn index_mut(&mut self, rreg: RealReg) -> &mut Self::Output {
    &mut self.regs[rreg.get_index() - self.offset]
  }
}

impl Registers {
  fn new(i: usize, reg_universe: &RealRegUniverse) -> Self {
    let mut regs = Vec::new();
    let mut offset = 0;
    // Collect all the registers for the current class.
    if let Some(ref range) = reg_universe.allocable_by_class[i] {
      debug_assert!(range.0 <= range.1);
      offset = range.0;
      for reg in &reg_universe.regs[range.0..=range.1] {
        debug_assert!(regs.len() == reg.0.get_index() - offset);
        regs.push((reg.0, true));
      }
    };
    Self { offset, regs }
  }

  fn is_taken(&self, reg: &RealReg) -> bool {
    !self[*reg].1
  }
  fn all_taken(&self) -> bool {
    !self.regs.iter().any(|reg| reg.1)
  }

  fn take_any(&mut self) -> RealReg {
    debug_assert!(!self.all_taken());
    for reg in self.regs.iter_mut() {
      if reg.1 {
        reg.1 = false;
        return reg.0;
      }
    }
    unreachable!();
  }
  fn take(&mut self, reg: &RealReg) {
    debug_assert!(self.regs[reg.get_index()].1, "taking an already taken reg");
    self[*reg].1 = false;
  }
  fn free(&mut self, reg: &RealReg) {
    debug_assert!(
      !self.regs[reg.get_index()].1,
      "freeing an already freed reg"
    );
    self[*reg].1 = true;
  }
}

struct Intervals<'a> {
  data: Vec<LiveInterval<'a>>,
}

impl<'a> Intervals<'a> {
  fn new(capacity: usize) -> Self {
    Self { data: Vec::with_capacity(capacity) }
  }
  fn push_real(&mut self, range: &'a mut RealRange) {
    let id = LiveId(self.data.len());
    self.data.push(LiveInterval::from_real(id, range))
  }
  fn push_virtual(&mut self, range: &'a mut VirtualRange) {
    let id = LiveId(self.data.len());
    self.data.push(LiveInterval::from_virtual(id, range))
  }
}

impl<'a> std::ops::Index<LiveId> for Intervals<'a> {
  type Output = LiveInterval<'a>;
  fn index(&self, id: LiveId) -> &Self::Output {
    &self.data[id.0]
  }
}

impl<'a> std::ops::IndexMut<LiveId> for Intervals<'a> {
  fn index_mut(&mut self, id: LiveId) -> &mut Self::Output {
    &mut self.data[id.0]
  }
}

/// State structure, which can be cleared between different calls to register allocation.
struct State<'a> {
  intervals: Intervals<'a>,

  /// A list of active intervals, sorted by increasing end point.
  active: Vec<LiveId>,

  /// A list of inactive intervals, sorted by increasing end point too.
  inactive: Vec<LiveId>,

  /// Unhandled intervals need to be processed by the main iteration loop,
  /// either because they have never been processed, or because they are
  /// inactive with another fragment.
  unhandled: Vec<LiveId>,

  /// Registers picker.
  regs: Vec<Registers>,
}

impl<'a> State<'a> {
  fn new(intervals: Intervals<'a>, reg_universe: &RealRegUniverse) -> Self {
    let mut all_regs = Vec::with_capacity(NUM_REG_CLASSES);
    for cur_reg_class in &[RegClass::I32, RegClass::F32] {
      all_regs.push(Registers::new(*cur_reg_class as usize, reg_universe))
    }
    let unhandled = intervals.data.iter().map(|int| int.id).collect();
    Self {
      intervals,
      unhandled,
      active: Vec::new(),
      inactive: Vec::new(),
      regs: all_regs,
    }
  }

  #[allow(dead_code)]
  pub fn clear(&mut self) {
    self.active.clear();
    self.inactive.clear();
    self.regs.clear();
  }

  fn regs(&mut self, reg_class: RegClass) -> &mut Registers {
    &mut self.regs[reg_class as usize]
  }

  fn next_unhandled(&mut self) -> Option<LiveId> {
    self.unhandled.first().cloned()
  }
}

fn insert_unhandled<'a>(
  unhandled: &mut Vec<LiveId>, intervals: &Intervals,
  interval: &LiveInterval<'a>, fragments: &Fragments,
) {
  // Add this interval to the list of active intervals.
  let index = unhandled
    .binary_search_by_key(&interval.start_point(&fragments), |unhandled_int| {
      intervals[*unhandled_int].start_point(&fragments)
    });
  let index = match index {
    Ok(index) => index,
    Err(index) => index,
  };
  unhandled.insert(index, interval.id);
}

fn insert_active<'a>(
  active: &mut Vec<LiveId>, intervals: &Intervals, interval: &LiveInterval<'a>,
  fragments: &Fragments,
) {
  // Add this interval to the list of active intervals.
  let index = active
    .binary_search_by_key(&interval.last_end_point(&fragments), |active_int| {
      intervals[*active_int].last_end_point(&fragments)
    });
  let index = match index {
    Ok(index) => index,
    Err(index) => index,
  };
  active.insert(index, interval.id);
}

// Allocator top level.  |func| is modified so that, when this function
// returns, it will contain no VirtualReg uses.  Allocation can fail if there
// are insufficient registers to even generate spill/reload code, or if the
// function appears to have any undefined VirtualReg/RealReg uses.
#[inline(never)]
pub fn alloc_main(
  func: &mut Func, reg_universe: &RealRegUniverse,
) -> Result<(), String> {
  let (mut rlrs, mut vlrs, fragments) = run_analysis(func)?;

  let intervals = {
    let mut int = Intervals::new(rlrs.len() as usize + vlrs.len() as usize);
    for rlr in rlrs.iter_mut() {
      int.push_real(rlr);
    }
    for vlr in vlrs.iter_mut() {
      int.push_virtual(vlr)
    }

    // Sort by increasing start point of their first fragment, since their other fragments are
    // already sorted.
    int.data.sort_by_key(|live_int| live_int.start_point(&fragments));

    int
  };

  let mut state = State::new(intervals, reg_universe);

  while let Some(interval_id) = state.next_unhandled() {
    let start_point = state.intervals[interval_id].start_point(&fragments);

    state = update_state(start_point, state, &fragments);

    let interval = &state.intervals[interval_id];
    debug!("handling {:?}", interval);
    let free_regs = &mut state.regs[interval.reg_class() as usize];

    let has_fixed_conflict = if let Some(fixed_reg) = interval.fixed_reg() {
      free_regs.is_taken(&fixed_reg)
    } else {
      false
    };

    if free_regs.all_taken() || has_fixed_conflict {
      spill();
    } else {
      if let Some(fixed_reg) = interval.fixed_reg() {
        // Mark fixed register as taken.
        free_regs.take(&fixed_reg);
      } else {
        // Pick any register and assign it.
        let reg = free_regs.take_any();
        insert_active(
          &mut state.active,
          &state.intervals,
          interval,
          &fragments,
        );

        let interval = &mut state.intervals[interval_id];
        interval.set_reg(reg);
      }
    }
  }

  reconcile_code(reg_universe, &vlrs, &fragments, func);

  Ok(())
}

fn reconcile_code(
  reg_universe: &RealRegUniverse, vlr_env: &VirtualRanges,
  frag_env: &Fragments, func: &mut Func,
) {
  // TODO copied from Julian's code.
  type PerRReg = Vec<VirtualRangeIx>;

  // Whereas this is empty.  We have to populate it "by hand", by
  // effectively cloning the allocatable part (prefix) of the universe.
  let mut perRReg = Vec::<PerRReg>::new();
  for _rreg in 0..reg_universe.allocable {
    // Doing this instead of simply .resize avoids needing Clone for PerRReg
    perRReg.push(PerRReg::new());
  }

  for (i, vlr) in vlr_env.iter().enumerate() {
    let rregNo = vlr.rreg.unwrap().get_index();
    let curr_vlrix = mkVirtualRangeIx(i as u32);
    perRReg[rregNo].push(curr_vlrix);
  }

  // -------- Edit the instruction stream --------

  // Gather up a vector of (RangeFrag, VirtualReg, RealReg) resulting from the previous
  // phase.  This fundamentally is the result of the allocation and tells us
  // how the instruction stream must be edited.  Note it does not take
  // account of spill or reload instructions.  Dealing with those is
  // relatively simple and happens later.
  //
  // Make two copies of this list, one sorted by the fragment start points
  // (just the InsnIx numbers, ignoring the Point), and one sorted by
  // fragment end points.

  let mut fragMapsByStart = Vec::<(RangeFragIx, VirtualReg, RealReg)>::new();
  let mut fragMapsByEnd = Vec::<(RangeFragIx, VirtualReg, RealReg)>::new();

  // For each real register ..
  // For each real register under our control ..
  for i in 0..reg_universe.allocable {
    let rreg = reg_universe.regs[i].0;
    // .. look at all the VLRs assigned to it.  And for each such VLR ..
    for vlrix_assigned in &perRReg[i] {
      let vlr_assigned = &vlr_env[*vlrix_assigned];
      // All the Frags in |vlr_assigned| require |vlr_assigned.reg| to
      // be mapped to the real reg |i|
      let vreg = vlr_assigned.vreg;
      // .. collect up all its constituent Frags.
      for fix in &vlr_assigned.sortedFrags.fragIxs {
        fragMapsByStart.push((*fix, vreg, rreg));
        fragMapsByEnd.push((*fix, vreg, rreg));
      }
    }
  }

  fragMapsByStart.sort_unstable_by(|(fixNo1, _, _), (fixNo2, _, _)| {
    frag_env[*fixNo1]
      .first
      .iix
      .partial_cmp(&frag_env[*fixNo2].first.iix)
      .unwrap()
  });

  fragMapsByEnd.sort_unstable_by(|(fixNo1, _, _), (fixNo2, _, _)| {
    frag_env[*fixNo1].last.iix.partial_cmp(&frag_env[*fixNo2].last.iix).unwrap()
  });

  //println!("Firsts: {}", fragMapsByStart.show());
  //println!("Lasts:  {}", fragMapsByEnd.show());

  let mut cursor_start = 0;
  #[allow(unused_assignments)]
  let mut num_start = 0;
  let mut cursor_ends = 0;
  #[allow(unused_assignments)]
  let mut num_ends = 0;

  let mut map = Map::<VirtualReg, RealReg>::default();

  fn is_sane(frag: &RangeFrag) -> bool {
    // "Normal" frag (unrelated to spilling).  No normal frag may start or
    // end at a .s or a .r point.
    if frag.first.pt.isUseOrDef()
      && frag.last.pt.isUseOrDef()
      && frag.first.iix <= frag.last.iix
    {
      return true;
    }
    // A spill-related ("bridge") frag.  There are three possibilities,
    // and they correspond exactly to |BridgeKind|.
    if frag.first.pt.isReload()
      && frag.last.pt.isUse()
      && frag.last.iix == frag.first.iix
    {
      // BridgeKind::RtoU
      return true;
    }
    if frag.first.pt.isReload()
      && frag.last.pt.isSpill()
      && frag.last.iix == frag.first.iix
    {
      // BridgeKind::RtoS
      return true;
    }
    if frag.first.pt.isDef()
      && frag.last.pt.isSpill()
      && frag.last.iix == frag.first.iix
    {
      // BridgeKind::DtoS
      return true;
    }
    // None of the above apply.  This RangeFrag is insane \o/
    false
  }

  for insnIx in mkInstIx(0).dotdot(mkInstIx(func.insns.len())) {
    //println!("");
    //println!("QQQQ insn {}: {}",
    //         insnIx, func.insns[insnIx].show());
    //println!("QQQQ init map {}", showMap(&map));
    // advance [cursor_start, +num_start) to the group for insnIx
    while cursor_start < fragMapsByStart.len()
      && frag_env[fragMapsByStart[cursor_start].0].first.iix < insnIx
    {
      cursor_start += 1;
    }
    num_start = 0;
    while cursor_start + num_start < fragMapsByStart.len()
      && frag_env[fragMapsByStart[cursor_start + num_start].0].first.iix
        == insnIx
    {
      num_start += 1;
    }

    // advance [cursor_ends, +num_ends) to the group for insnIx
    while cursor_ends < fragMapsByEnd.len()
      && frag_env[fragMapsByEnd[cursor_ends].0].last.iix < insnIx
    {
      cursor_ends += 1;
    }
    num_ends = 0;
    while cursor_ends + num_ends < fragMapsByEnd.len()
      && frag_env[fragMapsByEnd[cursor_ends + num_ends].0].last.iix == insnIx
    {
      num_ends += 1;
    }

    // So now, fragMapsByStart[cursor_start, +num_start) are the mappings
    // for fragments that begin at this instruction, in no particular
    // order.  And fragMapsByEnd[cursorEnd, +numEnd) are the FragIxs for
    // fragments that end at this instruction.

    //println!("insn no {}:", insnIx);
    //for j in cursor_start .. cursor_start + num_start {
    //    println!("   s: {} {}",
    //             (fragMapsByStart[j].1, fragMapsByStart[j].2).show(),
    //             frag_env[ fragMapsByStart[j].0 ]
    //             .show());
    //}
    //for j in cursor_ends .. cursor_ends + num_ends {
    //    println!("   e: {} {}",
    //             (fragMapsByEnd[j].1, fragMapsByEnd[j].2).show(),
    //             frag_env[ fragMapsByEnd[j].0 ]
    //             .show());
    //}

    // Sanity check all frags.  In particular, reload and spill frags are
    // heavily constrained.  No functional effect.
    for j in cursor_start..cursor_start + num_start {
      let frag = &frag_env[fragMapsByStart[j].0];
      // "It really starts here, as claimed."
      debug_assert!(frag.first.iix == insnIx);
      debug_assert!(is_sane(&frag));
    }
    for j in cursor_ends..cursor_ends + num_ends {
      let frag = &frag_env[fragMapsByEnd[j].0];
      // "It really ends here, as claimed."
      debug_assert!(frag.last.iix == insnIx);
      debug_assert!(is_sane(frag));
    }

    // Here's the plan, in summary:
    // Update map for I.r:
    //   add frags starting at I.r
    //   no frags should end at I.r (it's a reload insn)
    // Update map for I.u:
    //   add frags starting at I.u
    //   mapU := map
    //   remove frags ending at I.u
    // Update map for I.d:
    //   add frags starting at I.d
    //   mapD := map
    //   remove frags ending at I.d
    // Update map for I.s:
    //   no frags should start at I.s (it's a spill insn)
    //   remove frags ending at I.s
    // apply mapU/mapD to I

    // Update map for I.r:
    //   add frags starting at I.r
    //   no frags should end at I.r (it's a reload insn)
    for j in cursor_start..cursor_start + num_start {
      let frag = &frag_env[fragMapsByStart[j].0];
      if frag.first.pt.isReload() {
        //////// STARTS at I.r
        map.insert(fragMapsByStart[j].1, fragMapsByStart[j].2);
      }
    }

    // Update map for I.u:
    //   add frags starting at I.u
    //   mapU := map
    //   remove frags ending at I.u
    for j in cursor_start..cursor_start + num_start {
      let frag = &frag_env[fragMapsByStart[j].0];
      if frag.first.pt.isUse() {
        //////// STARTS at I.u
        map.insert(fragMapsByStart[j].1, fragMapsByStart[j].2);
      }
    }
    let mapU = map.clone();
    for j in cursor_ends..cursor_ends + num_ends {
      let frag = &frag_env[fragMapsByEnd[j].0];
      if frag.last.pt.isUse() {
        //////// ENDS at I.U
        map.remove(&fragMapsByEnd[j].1);
      }
    }

    // Update map for I.d:
    //   add frags starting at I.d
    //   mapD := map
    //   remove frags ending at I.d
    for j in cursor_start..cursor_start + num_start {
      let frag = &frag_env[fragMapsByStart[j].0];
      if frag.first.pt.isDef() {
        //////// STARTS at I.d
        map.insert(fragMapsByStart[j].1, fragMapsByStart[j].2);
      }
    }
    let mapD = map.clone();
    for j in cursor_ends..cursor_ends + num_ends {
      let frag = &frag_env[fragMapsByEnd[j].0];
      if frag.last.pt.isDef() {
        //////// ENDS at I.d
        map.remove(&fragMapsByEnd[j].1);
      }
    }

    // Update map for I.s:
    //   no frags should start at I.s (it's a spill insn)
    //   remove frags ending at I.s
    for j in cursor_ends..cursor_ends + num_ends {
      let frag = &frag_env[fragMapsByEnd[j].0];
      if frag.last.pt.isSpill() {
        //////// ENDS at I.s
        map.remove(&fragMapsByEnd[j].1);
      }
    }

    //println!("QQQQ mapU {}", showMap(&mapU));
    //println!("QQQQ mapD {}", showMap(&mapD));

    // Finally, we have mapU/mapD set correctly for this instruction.
    // Apply it.
    func.insns[insnIx].mapRegs_D_U(&mapD, &mapU);

    // Update cursor_start and cursor_ends for the next iteration
    cursor_start += num_start;
    cursor_ends += num_ends;

    if func.blocks.iter().any(|b| b.start.plus(b.len).minus(1) == insnIx) {
      //println!("Block end");
      debug_assert!(map.is_empty());
    }
  }

  debug_assert!(map.is_empty());
}

mod test_utils {
  use super::*;

  use crate::data_structures::make_universe;
  use crate::tests::find_Func;
  use crate::RunStage;

  pub fn lsra(func_name: &str, num_gpr: usize, num_fpu: usize) {
    let _ = pretty_env_logger::try_init();
    let mut func = find_Func(func_name).unwrap();
    let reg_universe = make_universe(num_gpr, num_fpu);
    func.run("Before allocation", &reg_universe, RunStage::BeforeRegalloc);
    alloc_main(&mut func, &reg_universe).unwrap_or_else(|err| {
      panic!("allocation failed: {}", err);
    });
    func.run("After allocation", &reg_universe, RunStage::AfterRegalloc);
  }
}

#[test]
fn lsra_badness() {
  test_utils::lsra("badness", 1, 0);
}

#[test]
fn lsra_straight_line() {
  test_utils::lsra("straight_line", 1, 0);
}

#[test]
fn lsra_fill_then_sum() {
  // TODO implement spill
  //test_utils::lsra("fill_then_sum", 32, 32);
  //test_utils::lsra("fill_then_sum", 8, 8);
}

#[test]
fn lsra_ssort() {
  test_utils::lsra("ssort", 10, 10);
  //test_utils::lsra("ssort", 8, 8);
}

#[test]
fn lsra_3_loops() {
  test_utils::lsra("3_loops", 8, 8);
}

#[test]
fn lsra_stmts() {
  test_utils::lsra("stmts", 8, 8);
}

#[test]
fn lsra_needs_splitting() {
  test_utils::lsra("needs_splitting", 8, 8);
}

#[test]
fn lsra_needs_splitting2() {
  test_utils::lsra("needs_splitting2", 8, 8);
}

#[test]
fn lsra_qsort() {
  test_utils::lsra("qsort", 8, 8);
}

#[test]
fn lsra_2a_fill_then_sum() {
  test_utils::lsra("fill_then_sum_2a", 8, 8);
}

#[test]
fn lsra_2a_ssort() {
  test_utils::lsra("ssort_2a", 8, 8);
}
