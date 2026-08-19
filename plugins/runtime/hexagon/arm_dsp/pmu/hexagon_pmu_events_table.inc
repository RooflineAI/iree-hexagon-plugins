
  // 0x0001: Can be used as the event detected by counter1 to build an effective
  // 64-bit counter. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COUNTER0_OVERFLOW, 0x0001)


  // 0x0002: Can be used as the event detected by counter3 to build an effective
  // 64-bit counter. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COUNTER2_OVERFLOW, 0x0002)


  // 0x0003: Number of packets that are committed by any thread. Packets are
  // executed. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_ANY, 0x0003)


  // 0x0004: Number of packets that are committed two cycles after an earlier
  // packet in the same thread. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_BSB, 0x0004)


  // 0x0005: Can be used as the event detected by counter5 to build an effective
  // 64-bit counter. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COUNTER4_OVERFLOW, 0x0005)


  // 0x0006: Can be used as the event detected by counter7 to build an effective
  // 64-bit counter. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COUNTER6_OVERFLOW, 0x0006)


  // 0x0007: Number of packets that are committed one cycle after the earlier
  // packet in the same thread. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_B2B, 0x0007)


  // 0x0008: Number of packets committed on the SMT threads (2nd/3rd/4th packets
  // in one cycle). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_SMT, 0x0008)


  // 0x000A: Processor cycles that exactly five threads are running (not
  // Wait/Stop). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CYCLES_5_THREAD_RUNNING, 0x000A)


  // 0x000B: Processor cycles that exactly six threads are running (not
  // Wait/Stop). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CYCLES_6_THREAD_RUNNING, 0x000B)


  // 0x000C: Number of packets committed by thread 0. Packets are executed. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_T0, 0x000C)


  // 0x000D: Number of packets committed by thread 1. Packets are executed. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_T1, 0x000D)


  // 0x000E: Number of packets committed by thread 2. Packets are executed. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_T2, 0x000E)


  // 0x000F: Number of packets committed by thread 3. Packets are executed. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_T3, 0x000F)


  // 0x0010: Number of packets committed by thread 4. Packets are executed. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_T4, 0x0010)


  // 0x0011: Number of packets committed by thread 5. Packets are executed. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_T5, 0x0011)


  // 0x0012: I-cache cacheable demand primary/secondary misses (includes
  // secondary misses). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_ICACHE_DEMAND_MISS, 0x0012)


  // 0x0013: D-cache cacheable demand primary/secondary misses; includes dczero
  // stalls; excludes uncacheables/prefetches/no-alloc store misses. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DCACHE_DEMAND_MISS, 0x0013)


  // 0x0014: D-cache cacheable store misses. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DCACHE_STORE_MISS, 0x0014)


  // 0x0015: Thread 6 committed a packet. Packets are executed. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_T6, 0x0015)


  // 0x0016: Thread 7 committed a packet. Packets are executed. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_T7, 0x0016)


  // 0x0017: Packets ready at CU scheduler but not scheduled (thread not picked
  // or inter-cluster resource conflict). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CU_PKT_READY_NOT_DISPATCHED, 0x0017)


  // 0x0018: Committed packets with five threads running (not Wait/Stop).
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_5_THREAD_RUNNING, 0x0018)


  // 0x0019: Committed packets with six threads running (not Wait/Stop).
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_6_THREAD_RUNNING, 0x0019)


  // 0x001A: Committed packets with seven threads running (not Wait/Stop).
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_7_THREAD_RUNNING, 0x001A)


  // 0x001B: Committed packets with eight threads running (not Wait/Stop).
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_8_THREAD_RUNNING, 0x001B)


  // 0x001C: IU L1S loads (demands or prefetches). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_IU_L1S_ACCESS, 0x001C)


  // 0x001D: IU L1S prefetches. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_IU_L1S_PREFETCH, 0x001D)


  // 0x001E: IU L1S stalls due to an AXI slave. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_IU_L1S_AXIS_STALL, 0x001E)


  // 0x001F: IU request to L1S and no grant from the vector unit. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_IU_L1S_NO_GRANT, 0x001F)


  // 0x0020: Any IU stall other than I-cache miss; includes jump register stall,
  // fetchcross, ITLB miss, etc; excludes CU replay. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_ANY_IU_REPLAY, 0x0020)


  // 0x0021: Any DU replay; includes bank conflict, store buffer full, etc;
  // excludes stall due to cache miss. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_ANY_DU_REPLAY, 0x0021)


  // 0x0022: Processor cycles that exactly seven threads are running (not
  // Wait/Stop). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CYCLES_7_THREAD_RUNNING, 0x0022)


  // 0x0023: Speculatively issued packets delivered from an IU. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_ISSUED_PACKETS, 0x0023)


  // 0x0025: Committed packets with one thread running (not Wait/Stop).
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_1_THREAD_RUNNING, 0x0025)


  // 0x0026: Committed packets with two threads running (not Wait/Stop).
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_2_THREAD_RUNNING, 0x0026)


  // 0x0027: Committed packets with three threads running (not Wait/Stop).
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_3_THREAD_RUNNING, 0x0027)


  // 0x0028: For a specific thread, sustained power exceeds LMH thresholds;
  // throttling based on thread priority. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_THREAD_LMH_THROTTLE, 0x0028)


  // 0x0029: Throttling based on peak current over LMH current limits. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_LMH_THROTTLE, 0x0029)


  // 0x002A: Committed instructions; up to 8/cycle; duplex counts as 2; excludes
  // end loops. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_INSTS, 0x002A)


  // 0x002B: Committed TC1 class instructions; duplex counts as 2; excludes
  // NOPs. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_TC1_INSTS, 0x002B)


  // 0x002C: Committed instructions using per-cluster (private) execution
  // resources; duplex counts as 2. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PRIVATE_INSTS, 0x002C)


  // 0x002D: Sustained global power exceeds thresholds; causes thread-specific
  // LMH engagement. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_GLOBAL_POWERLIMITS_OVER, 0x002D)


  // 0x002E: Processor cycles that exactly eight threads are running (not
  // Wait/Stop). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CYCLES_8_THREAD_RUNNING, 0x002E)


  // 0x002F: Committed packets with four threads running (not Stop/Wait).
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_4_THREAD_RUNNING, 0x002F)


  // 0x0030: Committed load instructions (cached/uncached); +2 for dual loads;
  // excludes prefetches/memops/coprocessor loads. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_LOADS, 0x0030)


  // 0x0031: Committed store instructions (cached/uncached); +2 for dual stores;
  // excludes memops/coprocessor stores. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_STORES, 0x0031)


  // 0x0032: Committed memory operations instructions (cached or uncached).
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_MEMOPS, 0x0032)


  // 0x0033: Committed NOPs. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_NOPS, 0x0033)


  // 0x0034: Speculatively issued instructions delivered from the IU. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_ISSUED_INSTS, 0x0034)


  // 0x0035: Packets that the CU dispatched; NOPs are squashed. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DISPATCHED_PACKETS, 0x0035)


  // 0x0036: Instructions that the CU dispatched. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DISPATCHED_INSTS, 0x0036)


  // 0x0037: Committed packets containing a program-flow instruction (CR jumps,
  // endloop, J, JR, dealloc_return, system/trap, ...). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PROGRAM_FLOW_INSTS, 0x0037)


  // 0x0038: Committed packets that changed control flow (taken jumps; includes
  // endloop & dealloc_return). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_CHANGED_FLOW, 0x0038)


  // 0x0039: Committed packets containing a taken end loop. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_ENDLOOP, 0x0039)


  // 0x003A: Store port used when p0 and p1 both occupied (only increments when
  // store port present). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_PST_USED_P0P1BUSY, 0x003A)


  // 0x003B: Processor cycles that exactly one thread is running (not
  // Wait/Stop). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CYCLES_1_THREAD_RUNNING, 0x003B)


  // 0x003C: Processor cycles that exactly two threads are running (not
  // Wait/Stop). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CYCLES_2_THREAD_RUNNING, 0x003C)


  // 0x003D: Processor cycles that exactly three threads are running (not
  // Wait/Stop). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CYCLES_3_THREAD_RUNNING, 0x003D)


  // 0x003E: Processor cycles that exactly four threads are running (not
  // Wait/Stop). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CYCLES_4_THREAD_RUNNING, 0x003E)


  // 0x003F: 128-byte line read requests issued by the primary AXI master. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_LINE128_READ_REQUEST, 0x003F)


  // 0x0040: All read requests issued by the primary AXI master
  // (full/partial/interleaved). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_READ_REQUEST, 0x0040)


  // 0x0041: 32-byte line read requests issued by the primary AXI master. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_LINE32_READ_REQUEST, 0x0041)


  // 0x0042: All write requests issued by the primary AXI master
  // (full/partial/interleaved). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_WRITE_REQUEST, 0x0042)


  // 0x0043: 32-byte line write requests issued by the primary AXI master; all
  // bytes valid. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_LINE32_WRITE_REQUEST, 0x0043)


  // 0x0044: Read requests issued by the AHB master. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AHB_READ_REQUEST, 0x0044)


  // 0x0045: Write requests issued by the AHB master. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AHB_WRITE_REQUEST, 0x0045)


  // 0x0046: 128-byte line write requests issued by the primary AXI master; all
  // bytes valid. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_LINE128_WRITE_REQUEST, 0x0046)


  // 0x0047: AXI slave multi-beat accesses. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_SLAVE_MULTI_BEAT_ACCESS, 0x0047)


  // 0x0048: AXI slave single-beat accesses. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_SLAVE_SINGLE_BEAT_ACCESS, 0x0048)


  // 0x0049: All read requests issued by the secondary AXI master
  // (full/partial). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI2_READ_REQUEST, 0x0049)


  // 0x004A: 32-byte line read requests issued by the secondary AXI master. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI2_LINE32_READ_REQUEST, 0x004A)


  // 0x004B: All write requests issued by the secondary AXI master
  // (full/partial). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI2_WRITE_REQUEST, 0x004B)


  // 0x004C: 32-byte line write requests issued by the secondary AXI master.
  // (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI2_LINE32_WRITE_REQUEST, 0x004C)


  // 0x004D: Secondary AXI command/data queue full; operation stuck at head of
  // command queue. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI2_CONGESTION, 0x004D)


  // 0x0050: Committed floating point instructions; +2 for dual FP ops; excludes
  // conversions. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_FPS, 0x0050)


  // 0x0051: Mispredicted bimodal branch direction caused a redirect. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_REDIRECT_BIMODAL_MISPREDICT, 0x0051)


  // 0x0052: Mispredicted branch target caused a redirect; includes RAS
  // mispredict/HintJR mispredict; excludes most indirects. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_REDIRECT_TARGET_MISPREDICT, 0x0052)


  // 0x0053: Mispredicted hardware loop end caused a redirect (small loops, low
  // counts). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_REDIRECT_LOOP_MISPREDICT, 0x0053)


  // 0x0054: Redirect for reasons other than 0x51-0x53 (exceptions, traps,
  // interrupts, etc.). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_REDIRECT_MISC, 0x0054)


  // 0x0055: 256-byte line write requests issued by the AXI master; all bytes
  // valid. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_LINE256_WRITE_REQUEST, 0x0055)


  // 0x0056: Number of packets that cracked. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_NUM_PACKET_CRACKED, 0x0056)


  // 0x0058: JTLB miss for instruction or data translation request. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_JTLB_MISS, 0x0058)


  // 0x005A: Committed return instructions (includes canceled returns).
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_RETURN, 0x005A)


  // 0x005B: Committed indirect jumps/calls (includes canceled; excludes JUMPR
  // R31 returns). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_INDIRECT_JUMP, 0x005B)


  // 0x005C: Committed bimodal branches (*.old/*.new); +2 for dual jumps.
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_BIMODAL_BRANCH_INSTS, 0x005C)


  // 0x005F: Cycles cluster can be issued if VTCM FIFO queue is full. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_VTCM_FIFO_FULL_CYCLES, 0x005F)


  // 0x0061: Scalar load accesses to L1S. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_L1S_LOAD_ACCESS, 0x0061)


  // 0x0062: I-cache line fetches. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_ICACHE_ACCESS, 0x0062)


  // 0x0063: Branch target buffer hits. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_BTB_HIT, 0x0063)


  // 0x0064: Branch target buffer misses. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_BTB_MISS, 0x0064)


  // 0x0065: I-cache secondary misses. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_IU_DEMAND_SECONDARY_MISS, 0x0065)


  // 0x0067: Fast fetches killed (after an I-cache access). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_FAST_FETCH_KILLED, 0x0067)


  // 0x0069: Packets dropped because IU cannot deliver them to CU. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_FETCHED_PACKETS_DROPPED, 0x0069)


  // 0x006B: IU prefetches sent to L2; counts accepted prefetches; excludes
  // replayed; excludes IU->ITCM. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_IU_PREFETCHES_SENT_TO_L2, 0x006B)


  // 0x006C: ITLB misses that go to JTLB. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_ITLB_MISS, 0x006C)


  // 0x0072: Two-cycle fetches in IU (returns, loop end, fall through, BTB).
  // (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_FETCH_2_CYCLE, 0x0072)


  // 0x0073: Three-cycle fetches in IU. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_FETCH_3_CYCLE, 0x0073)


  // 0x0075: L2 secondary misses from an IU. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_IU_SECONDARY_MISS, 0x0075)


  // 0x0076: L2 cacheable accesses from IU (demand/L1 prefetch); excludes L2
  // prefetch-gen, L2fetch, TCM, uncacheables; targets primary AXI. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_IU_ACCESS, 0x0076)


  // 0x0077: L2 misses from IU (subset of 0x76 that missed); excludes pipeline
  // conflicts. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_IU_MISS, 0x0077)


  // 0x0078: IU prefetch accesses to L2; excludes L2fetch-generated; targets
  // primary AXI. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_IU_PREFETCH_ACCESS, 0x0078)


  // 0x0079: L2 misses that were IU prefetches (subset of 0x78 that missed).
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_IU_PREFETCH_MISS, 0x0079)


  // 0x007A: Requests IU->L2 to write bimodal bits in L2/TCM (all requests).
  // (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_IU_BRANCH_CACHE_WRITE_REQUEST, 0x007A)


  // 0x007B: Writes of bimodal bits in L2/TCM that completed (from 0x7A
  // requests). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_IU_BRANCH_CACHE_WRITE, 0x007B)


  // 0x007C: L2 cacheable read accesses from DU (loads/L1 prefetches/dcfetches);
  // excludes initial L2fetch, uncacheables, TCM, coproc; targets primary AXI.
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_DU_READ_ACCESS, 0x007C)


  // 0x007D: L2 read misses from DU (subset of 0x7C that missed/line not
  // allocated). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_DU_READ_MISS, 0x007D)


  // 0x007E: L2 fetch accesses initiated by programming L2fetch engine.
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2FETCH_ACCESS, 0x007E)


  // 0x007F: L2 fetch misses from programmed inquiry (subset of 0x7E that
  // missed). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2FETCH_MISS, 0x007F)


  // 0x0081: All requests to L2 cache (includes L2Fetch programming; excludes
  // internally generated L2 fetch accesses); includes cacheable/TCM. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_ACCESS, 0x0081)


  // 0x0082: L2 request not taken due to pipe conflict (tag bank/data
  // bank/other). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_PIPE_CONFLICT_STALL, 0x0082)


  // 0x0083: Tag array conflicts (subset of 0x82/0x130 per table note).
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_TAG_ARRAY_CONFLICT, 0x0083)


  // 0x0087: TCM accesses from DU (L2 TCM space); excludes HVX requests.
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_TCM_DU_ACCESS, 0x0087)


  // 0x0088: TCM read accesses from DU (L2 TCM space); includes HVX requests.
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_TCM_DU_READ_ACCESS, 0x0088)


  // 0x0089: TCM accesses from IU (L2 TCM space). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_TCM_IU_ACCESS, 0x0089)


  // 0x008A: L2 dirty line evictions due to allocation (not on cache ops). (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_CASTOUT, 0x008A)


  // 0x008B: L2 cacheable store accesses from DU; excludes cache
  // ops/uncacheables/TCM/coproc stores; targets primary AXI. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_DU_STORE_ACCESS, 0x008B)


  // 0x008C: L2 store misses from DU (subset of 0x8B that missed: line not in
  // cache/coalesce buffer). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_DU_STORE_MISS, 0x008C)


  // 0x008D: L2 prefetch accesses from DU (dcfetch/dhwprefetch subset of 0x7C).
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_DU_PREFETCH_ACCESS, 0x008D)


  // 0x008E: L2 prefetch misses from DU (subset of 0x8D that missed). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_DU_PREFETCH_MISS, 0x008E)


  // 0x0090: L2 load secondary misses from DU (busy line in scoreboard prevented
  // return). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_DU_LOAD_SECONDARY_MISS, 0x0090)


  // 0x0091: L2fetch commands (excludes stop commands). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2FETCH_COMMAND, 0x0091)


  // 0x0092: L2fetch command killed due to stop command (one per killed
  // command). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2FETCH_COMMAND_KILLED, 0x0092)


  // 0x0093: L2fetch command overwritten (kills old and replaces with new). (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2FETCH_COMMAND_OVERWRITE, 0x0093)


  // 0x0094: L2fetch access blocked due to missing L2 fetch/evict credit. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2FETCH_ACCESS_CREDIT_FAIL, 0x0094)


  // 0x0095: AXI slave read access hit a busy line. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_SLAVE_READ_BUSY, 0x0095)


  // 0x0096: AXI slave write access hit a busy line. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_SLAVE_WRITE_BUSY, 0x0096)


  // 0x0097: Even-L2 accesses (subset of 0x81 to even interleave). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_ACCESS_EVEN, 0x0097)


  // 0x0098: High-priority CLADE IU/DU requests (not counted for L2 fetch).
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CLADE_HIGH_PRIO_L2_ACCESS, 0x0098)


  // 0x0099: Low-priority CLADE IU/DU requests (not counted for L2 fetch).
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CLADE_LOW_PRIO_L2_ACCESS, 0x0099)


  // 0x009A: CLADE high-priority L2 accesses that missed. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CLADE_HIGH_PRIO_L2_MISS, 0x009A)


  // 0x009B: CLADE low-priority L2 accesses that missed. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CLADE_LOW_PRIO_L2_MISS, 0x009B)


  // 0x009C: CLADE high-priority decode had an exception. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CLADE_HIGH_PRIO_EXCEPTION, 0x009C)


  // 0x009D: CLADE low-priority decode had an exception. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CLADE_LOW_PRIO_EXCEPTION, 0x009D)


  // 0x009E: AXI secondary slave read access hit a busy line. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI2_SLAVE_READ_BUSY, 0x009E)


  // 0x009F: AXI secondary slave write access hit a busy line. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI2_SLAVE_WRITE_BUSY, 0x009F)


  // 0x00A0: Any DU stall (D-cache miss or DTLB miss); increments once when
  // thread has DU stall. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_ANY_DU_STALL, 0x00A0)


  // 0x00A1: DU bank conflict replay. Dual memory access to same bank but
  // different lines. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_BANK_CONFLICT_REPLAY, 0x00A1)


  // 0x00A2: Packet replay due to insufficient QoS DU credits. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_CREDIT_REPLAY, 0x00A2)


  // 0x00A3: L2 even/odd FIFO full replays. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_FIFO_FULL_REPLAY, 0x00A3)


  // 0x00A4: DU replays because demand load hit in store buffer. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_STORE_BUFFER_FULL_REPLAY, 0x00A4)


  // 0x00A7: DU snoop requests accepted. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_SNOOP_REQUEST, 0x00A7)


  // 0x00A8: Fill replay due to index conflict with instruction in pipeline.
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_FILL_REPLAY, 0x00A8)


  // 0x00AC: DU reads to L2 (everything that brings data from L2 array; includes
  // prefetches; excludes coproc loads). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_READ_TO_L2, 0x00AC)


  // 0x00AD: DU writes to L2 (everything written out of DU to L2 array; includes
  // dczeroa; excludes cache ops/tag writes/coproc stores). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_WRITE_TO_L2, 0x00AD)


  // 0x00AF: dczeroa committed. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DCZERO_COMMITTED, 0x00AF)


  // 0x00B0: ITCM accesses from IU (IU demand fetches + IU prefetches); not
  // included in other L2 events. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2ITCM_IU_READ, 0x00B0)


  // 0x00B1: L2 ITCM read accesses from DU (demands + prefetches); not included
  // in other L2 events. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2ITCM_DU_READ, 0x00B1)


  // 0x00B2: L2 ITCM write accesses from DU (stores + dczeroa); excludes cache
  // ops; not included in other L2 events. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2ITCM_DU_WRITE, 0x00B2)


  // 0x00B3: DTLB miss that goes to JTLB; dual-slot behavior per description.
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DTLB_MISS, 0x00B3)


  // 0x00B4: Successful bimodal writes into L2 ITCM. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2ITCM_BIMODAL_WRITES_SUCCESS, 0x00B4)


  // 0x00B6: Store buffer hit replayed due to bank conflicts followed by load
  // from store buffer. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_STORE_BUFFER_HIT_REPLAY, 0x00B6)


  // 0x00B7: Store buffer must drain; forces packet replay (cache index match,
  // timeout, etc.). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_STORE_BUFFER_FORCE_REPLAY, 0x00B7)


  // 0x00B8: Inter-cluster tag write conflicts. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_TAG_WRITE_CONFLICT_REPLAY, 0x00B8)


  // 0x00B9: Inter-thread SMT bank conflicts. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_SMT_BANK_CONFLICT, 0x00B9)


  // 0x00BA: All port conflict replays (incl. same-cluster + inter-cluster).
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_PORT_CONFLICT_REPLAY, 0x00BA)


  // 0x00BB: Bimodal writes into L2 ITCM that were dropped. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2ITCM_BIMODAL_WRITES_DROPPED, 0x00BB)


  // 0x00BC: ITCM accesses from IU prefetches only; included in event 0xB0; not
  // in other L2 events. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2ITCM_IU_PREFETCH_READ, 0x00BC)


  // 0x00BD: Page cross from valid packet causing replay; counts twice if both
  // slots page-cross. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_PAGE_CROSS_REPLAY, 0x00BD)


  // 0x00BE: Slot0 store sent to other store buffer due to other cluster slot1
  // store/memop (store port present only). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_PST_STORE_SENTON_OTHPORT, 0x00BE)


  // 0x00BF: DU demand secondary misses. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_DEMAND_SECONDARY_MISS, 0x00BF)


  // 0x00C0: DU replays not counted by other replay events (when ANY_DU_REPLAY
  // counts and no other DU replay event counts). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_MISC_REPLAY, 0x00C0)


  // 0x00C2: Access replay because earlier access allocates/invalidates a way in
  // same index. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_STATE_REPLAY, 0x00C2)


  // 0x00C3: dcfetches committed (hits and drops; excludes
  // convert-to-prefetches). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DCFETCH_COMMITTED, 0x00C3)


  // 0x00C4: dcfetch hits in D-cache (valid or reserved lines). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DCFETCH_HIT, 0x00C4)


  // 0x00C5: dcfetches missed in L1; counts those issued to L2 FIFO. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DCFETCH_MISS, 0x00C5)


  // 0x00C8: Load instructions with uncacheable addresses in L1. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_LOAD_UNCACHEABLE, 0x00C8)


  // 0x00C9: Packets where both loads have uncacheable addresses in L1.
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_DUAL_LOAD_UNCACHEABLE, 0x00C9)


  // 0x00CA: Store instructions with uncacheable addresses in L1. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_STORE_UNCACHEABLE, 0x00CA)


  // 0x00CB: Stall due to insufficient credits from store release. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_STORE_RELEASE_CREDIT_STALL, 0x00CB)


  // 0x00CD: 256-byte line read requests issued by AXI master; all bytes valid.
  // (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_LINE256_READ_REQUEST, 0x00CD)


  // 0x00CE: 64-byte line read requests issued by primary AXI master. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_LINE64_READ_REQUEST, 0x00CE)


  // 0x00CF: 64-byte line write requests issued by primary AXI master; all bytes
  // valid. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_LINE64_WRITE_REQUEST, 0x00CF)


  // 0x00D1: 8-byte read requests issued by AHB. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AHB_8_READ_REQUEST, 0x00D1)


  // 0x00D3: L2fetch terminated due to VA->PA translation failure (incl.
  // permission errors). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2FETCH_COMMAND_PAGE_TERMINATION, 0x00D3)


  // 0x00D5: Of events in 0x8B (store access), number that were coalesced.
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_DU_STORE_COALESCE, 0x00D5)


  // 0x00D6: New store links to something else in scoreboard. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_STORE_LINK, 0x00D6)


  // 0x00D7: Cycles where L2 scoreboard >=70% full (e.g., 23+ of 32 entries);
  // uses most-consumed interleave. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_SCOREBOARD_70_PERCENT_FULL, 0x00D7)


  // 0x00D8: Cycles where L2 scoreboard >=80% full (e.g., 26+ of 32 entries);
  // uses most-consumed interleave. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_SCOREBOARD_80_PERCENT_FULL, 0x00D8)


  // 0x00D9: Cycles where L2 scoreboard >=90% full (e.g., 29+ of 32 entries);
  // uses most-consumed interleave. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_SCOREBOARD_90_PERCENT_FULL, 0x00D9)


  // 0x00DA: L2 scoreboard too full to accept selector request (and selector has
  // request). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_SCOREBOARD_FULL_REJECT, 0x00DA)


  // 0x00DC: Cycles when all eviction buffers in any interleave are occupied.
  // (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_EVICTION_BUFFERS_FULL, 0x00DC)


  // 0x00DD: 32-byte multi-beat read requests issued by AHB. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AHB_MULTI_BEAT_READ_REQUEST, 0x00DD)


  // 0x00DF: Of 0x90 events, those where primary miss was DC fetch or L2 fetch.
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_DU_LOAD_SECONDARY_MISS_ON_SW_PREFETCH, 0x00DF)


  // 0x00E0: L2 fetch data dropped because previous eviction not completed. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2FETCH_DROP, 0x00E0)


  // 0x00E5: Cycles cannot commit because thread is Off/Wait/pause. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_THREAD_IDLE_PVIEW_CYCLES, 0x00E5)


  // 0x00E6: Cycles cannot commit due to kernel lock or TLB lock. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_ARCH_LOCK_PVIEW_CYCLES, 0x00E6)


  // 0x00E7: Cycles cannot commit due to redirects (branch mispredicts, etc.).
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_REDIRECT_PVIEW_CYCLES, 0x00E7)


  // 0x00E8: Cycles cannot commit because issue queue is empty. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_IU_NO_PKT_PVIEW_CYCLES, 0x00E8)


  // 0x00E9: Cycles cannot commit due to D-cache cacheable miss. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_CACHE_MISS_PVIEW_CYCLES, 0x00E9)


  // 0x00EA: Cycles cannot commit due to DU replay/DU bubble/DTLB miss.
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_BUSY_OTHER_PVIEW_CYCLES, 0x00EA)


  // 0x00EB: Cycles cannot commit due to CU busy (interlocks/port
  // conflicts/timing class/HVX FIFO full/etc.). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CU_BUSY_PVIEW_CYCLES, 0x00EB)


  // 0x00EC: Cycles cannot commit due to DU resource conflict. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_CONFLICT_PVIEW_CYCLES, 0x00EC)


  // 0x00ED: Cycles cannot commit because coprocessor is busy. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COPROC_BUSY_PVIEW_CYCLES, 0x00ED)


  // 0x00EE: Cycles cannot commit due to D-cache uncacheable access. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_UNCACHED_PVIEW_CYCLES, 0x00EE)


  // 0x00EF: Cycles cannot commit due to system stalls (DMA sync/ETM full/Qtimer
  // read not ready/AXI busy/global cache op sync/etc.). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_SYSTEM_BUSY_PVIEW_CYCLES, 0x00EF)


  // 0x00F1: 128-byte line read requests issued by even-interleaved AXI master.
  // (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_LINE128_READ_REQUEST_EVEN, 0x00F1)


  // 0x00F2: All read requests issued by even-interleaved AXI master. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_READ_REQUEST_EVEN, 0x00F2)


  // 0x00F3: 32-byte line read requests issued by even-interleaved AXI master.
  // (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_LINE32_READ_REQUEST_EVEN, 0x00F3)


  // 0x00F4: All write requests issued by even-interleaved AXI master. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_WRITE_REQUEST_EVEN, 0x00F4)


  // 0x00F5: 32-byte line write requests issued by even-interleaved AXI master;
  // all bytes valid. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_LINE32_WRITE_REQUEST_EVEN, 0x00F5)


  // 0x00F6: 128-byte line write requests issued by even-interleaved AXI master;
  // all bytes valid. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_LINE128_WRITE_REQUEST_EVEN, 0x00F6)


  // 0x00F8: 64-byte line read requests issued by even-interleaved AXI master.
  // (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_LINE64_READ_REQUEST_EVEN, 0x00F8)


  // 0x00F9: 64-byte line write requests issued by even-interleaved AXI master;
  // all bytes valid. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_LINE64_WRITE_REQUEST_EVEN, 0x00F9)


  // 0x00FA: Even-interleaved AXI write command/data queue full; op stuck at
  // head of queue. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_WR_CONGESTION_EVEN, 0x00FA)


  // 0x00FB: Line-sized write to even-interleaved AXI master not all bytes valid
  // (includes segmented; excludes WT stores). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_INCOMPLETE_WRITE_REQUEST_EVEN, 0x00FB)


  // 0x00FC: 256-byte line read requests issued by even-interleaved AXI master;
  // all bytes valid. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_LINE256_READ_REQUEST_EVEN, 0x00FC)


  // 0x00FD: 256-byte line write requests issued by even-interleaved AXI master;
  // all bytes valid. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_LINE256_WRITE_REQUEST_EVEN, 0x00FD)


  // 0x00FE: Cycles where a cluster has 3 threads in Run mode with coprocessor
  // bit (SSR.XE) enabled. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CYCLES_3_COPROC_THREADS_ONE_CLUSTER, 0x00FE)


  // 0x02FA: Clean line evictions from L2 due to allocation; not on cache
  // operations. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_L2_CLEAN_CASTOUT, 0x02FA)


  // 0x02FB: All read requests issued by tertiary AXI master (full/partial).
  // (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI3_READ_REQUEST, 0x02FB)


  // 0x02FC: 32-byte line read requests issued by tertiary AXI master. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI3_LINE32_READ_REQUEST, 0x02FC)


  // 0x02FD: All write requests issued by tertiary AXI master (full/partial).
  // (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI3_WRITE_REQUEST, 0x02FD)


  // 0x02FE: 32-byte line write requests issued by tertiary AXI master. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI3_LINE32_WRITE_REQUEST, 0x02FE)


  // 0x02FF: Tertiary AXI read command queue full; op stuck at head of primary
  // AXI command queue (includes interleaved). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI3_RD_CONGESTION, 0x02FF)


  // 0x0300: Cycles when one packet is committed. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CYCLES_1_PACKET_COMMITTED, 0x0300)


  // 0x0301: Cycles when two packets are committed. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CYCLES_2_PACKET_COMMITTED, 0x0301)


  // 0x0302: Cycles when three packets are committed. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CYCLES_3_PACKET_COMMITTED, 0x0302)


  // 0x0303: Cycles when four packets are committed. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CYCLES_4_PACKET_COMMITTED, 0x0303)


  // 0x0304: Cycles when more than one packet is committed in cluster 0. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_SMT_CLUSTER0, 0x0304)


  // 0x0305: Cycles when more than one packet is committed in cluster 1. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_SMT_CLUSTER1, 0x0305)


  // 0x0306: Cycles when packets are committed on both clusters. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_SMT_INTERCLUSTER, 0x0306)


  // 0x0307: Packet is SMT-able w/o slot conflicts but cannot go due to s0/s1
  // reg read/CU forwarding. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_SMT_CONFLICT_FOR_REG_READ_OR_CU_FWD, 0x0307)


  // 0x0308: Committed packets with 2 threads running on same cluster (2T+0T).
  // (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_2_THREAD_RUNNING_2T_PLUS_0T, 0x0308)


  // 0x0309: Committed packets with 2 threads running on different clusters
  // (1T+1T). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_2_THREAD_RUNNING_1T_PLUS_1T, 0x0309)


  // 0x030A: Committed packets with 3 threads running on same cluster (3T+0T).
  // (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_3_THREAD_RUNNING_3T_PLUS_0T, 0x030A)


  // 0x030B: Committed packets with 3 threads running (2T+1T). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_3_THREAD_RUNNING_2T_PLUS_1T, 0x030B)


  // 0x030C: Committed packets with 4 threads running on same cluster (4T+0T).
  // (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_4_THREAD_RUNNING_4T_PLUS_0T, 0x030C)


  // 0x030D: Committed packets with 4 threads running (3T+1T). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_4_THREAD_RUNNING_3T_PLUS_1T, 0x030D)


  // 0x030E: Committed packets with 4 threads running (2T+2T). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_4_THREAD_RUNNING_2T_PLUS_2T, 0x030E)


  // 0x030F: Committed packets with 5 threads running (4T+1T). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_5_THREAD_RUNNING_4T_PLUS_1T, 0x030F)


  // 0x0310: Committed packets with 5 threads running (3T+2T). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_5_THREAD_RUNNING_3T_PLUS_2T, 0x0310)


  // 0x0311: Committed packets with 6 threads running (4T+2T). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_6_THREAD_RUNNING_4T_PLUS_2T, 0x0311)


  // 0x0312: Committed packets with 6 threads running (3T+3T). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_COMMITTED_PKT_6_THREAD_RUNNING_3T_PLUS_3T, 0x0312)


  // 0x0313: iprefetches initiated on demand misses. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_ICACHE_DEMAND_MISS_PREFETCH_MISS, 0x0313)


  // 0x0314: Committed simple packets dispatchable on in-cluster SMT threads
  // (eligible packets on primary+SMT). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_SIMPLE_PACKET, 0x0314)


  // 0x0315: 64-byte line write requests issued by primary AXI3 master; includes
  // interleaved; all bytes valid. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI3_LINE64_WRITE_REQUEST, 0x0315)


  // 0x0316: 64-byte line read requests issued by primary AXI3 master; includes
  // interleaved. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI3_LINE64_READ_REQUEST, 0x0316)


  // 0x0317: Tertiary AXI write command/data queue full; op stuck at head of
  // primary AXI3 command queue (includes interleaved). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI3_WR_CONGESTION, 0x0317)


  // 0x0318: Line-sized write to AXI3 master not all bytes valid (includes
  // segmented; excludes WT stores). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI3_INCOMPLETE_WRITE_REQUEST, 0x0318)


  // 0x0319: I-cache data replays due to incorrect way predictions. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_ICACHE_DATA_REPLAY, 0x0319)


  // 0x031C: In-cluster SMT thread picked, but not committed. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_SMT_PKT_PICKED_BUT_NOT_DISP, 0x031C)


  // 0x0322: CLADE2 both eviction buffer entries used (congested). (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CLADE2_EB_FULL, 0x0322)


  // 0x0323: L2 cache read requests in CLADE2 region. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CLADE2_RD_REQ, 0x0323)


  // 0x0324: L2 cache read request misses in CLADE2 region. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CLADE2_RDCACHE_MISS, 0x0324)


  // 0x0325: L2 cache write requests in CLADE2 region. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CLADE2_WR_REQ, 0x0325)


  // 0x0326: L2 cache write request misses in CLADE2 region. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CLADE2_WRCACHE_MISS, 0x0326)


  // 0x0327: L2 cache eviction of clean data to AXI bus. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_EWD_REQUEST, 0x0327)


  // 0x0328: L2 cache eviction of clean data to even-interleaved AXI bus. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_EWD_REQUEST_EVEN, 0x0328)


  // 0x0329: Cache maintenance operation request from QDSP6 core to AXIM. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_CMO_REQUEST, 0x0329)


  // 0x032A: Cache maintenance operation request from QDSP6 core to AXIM
  // Interleave0. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_AXI_CMO_REQUEST_EVEN, 0x032A)


  // 0x032D: Allocations in TAGE branch predictor due to mispredicting branches.
  // (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_TAGE_TABLE_ALLOC, 0x032D)


  // 0x032E: Hits in TAGE branch predictor table. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_TAGE_TABLE_HIT, 0x032E)


  // 0x032F: Times TAGE overrode bimodal prediction for a branch instruction.
  // (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_TAGE_BRANCH_OVERRIDE, 0x032F)


  // 0x0333: Stopping prefetching due to page cross. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_SPF_DTLBPGCROSS, 0x0333)


  // 0x0334: Prefetch requests that hit in L1D$. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_SPF_DCACHE_HIT, 0x0334)


  // 0x0335: Prefetch requests missing in L1D$. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_SPF_DCACHE_MISS, 0x0335)


  // 0x0336: Prefetch retry due to L2FIFO queue full. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_SPF_L2FIFOFULL_RETRY, 0x0336)


  // 0x0337: Prefetch retry due to L2 credits/AQoS busy. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_SPF_L2BUFFULL_RETRY, 0x0337)


  // 0x0338: Cycles prefetches losing arbitration. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_SPF_CONFLICT_RETRY, 0x0338)


  // 0x0340: Relative DPM value produced every cycle; see description for
  // converting to mW; value=1 indicates local DPM accumulator overflow event.
  // (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DPM_AVG_COMPRESSED, 0x0340)


  // 0x0351: DU replays due to way misprediction. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_DU_WAY_PRED_REPLAYS, 0x0351)


  // 0x0387: Dispatch blocks due to inter-cluster bank conflict prediction
  // (per-thread). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CU_BANK_CONFLICT_BLOCK, 0x0387)


  // 0x0388: Flushes due to incorrect inter-cluster bank conflict prediction
  // (per-thread). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CU_BANK_CONFLICT_FLUSH, 0x0388)


  // 0x0389: Valid bank conflict predictions for a given thread. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CU_BANK_CONFLICT_PREDICTION, 0x0389)


  // 0x0100: VFIFO not empty. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVX_ACTIVE, 0x0100)


  // 0x0101: Stall cycles due to interlocks. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVX_REG_ORDER, 0x0101)


  // 0x0102: Stall cycles due to accumulator not produced in previous context
  // cycle. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVX_ACC_ORDER, 0x0102)


  // 0x0103: Stall cycles due to load pending. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVX_LD_L2_OUTSTANDING, 0x0103)


  // 0x0104: Stall cycles due to store not yet allocated in L2. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVX_ST_L2_OUTSTANDING, 0x0104)


  // 0x0105: Stall cycles due to VTCM transaction pending. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVX_VTCM_OUTSTANDING, 0x0105)


  // 0x0106: Scatter/gather: Network scoreboard not updated. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVX_SCATGATH_FULL, 0x0106)


  // 0x0107: Scatter/gather input buffer full. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVX_SCATGATH_IN_FULL, 0x0107)


  // 0x0108: Store buffer full. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVX_ST_FULL, 0x0108)


  // 0x010A: Throttling: Voltage model would exceed undershoot threshold.
  // (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVX_VOLTAGE_UNDER, 0x010A)


  // 0x010B: Throttling: Sustained power exceeds budget. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVX_POWER_OVER, 0x010B)


  // 0x010C: Stall cycles due to multi-issue packet. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVX_PKT_PARTIAL, 0x010C)


  // 0x0111: Increments by 2 per packet in 128-byte mode; packets with HVX
  // instructions. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVX_PKT, 0x0111)


  // 0x0112: Committed packets on a thread with XE bit set (executed in Q6 or
  // coprocessor). (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVX_PKT_THREAD, 0x0112)


  // 0x0113: Cycles a thread stalled due to VFIFO. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVX_CORE_VFIFO_FULL_STALL, 0x0113)


  // 0x0115: Cycles 1 HVX context running. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CYCLES_1_HVX_CONTEXTS_RUNNING, 0x0115)


  // 0x0116: Cycles 2 HVX contexts running concurrently. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CYCLES_2_HVX_CONTEXTS_RUNNING, 0x0116)


  // 0x0117: Cycles 3 HVX contexts running concurrently. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CYCLES_3_HVX_CONTEXTS_RUNNING, 0x0117)


  // 0x0118: L2 cacheable load access from HVX; may cause L2 lookup; excludes
  // cache ops/uncacheables/scalars. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVXLD_L2, 0x0118)


  // 0x0119: TCM load access for HVX (HVX load from L2 TCM space). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVXLD_L2_TCM, 0x0119)


  // 0x011A: L2 cacheable miss from HVX load; subset of 0xFB-qualified events
  // that missed. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVXLD_L2_MISS, 0x011A)


  // 0x011B: HVX load secondary miss: load not returned due to prior access
  // being pending load or pending L2Fetch. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVXLD_L2_SECONDARY_MISS, 0x011B)


  // 0x011C: Vector write in L2. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVXST_L2_WR, 0x011C)


  // 0x011D: Lower priority vs scalar load for L2 return data bus. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVXST_SLD_CONFLICT, 0x011D)


  // 0x011E: Lower priority vs gather. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVXST_VTCM_GATH_CONFLICT, 0x011E)


  // 0x0121: Write FIFO full. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVXST_L2_FULL, 0x0121)


  // 0x0122: Write FIFO full. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVXST_VTCM_FULL, 0x0122)


  // 0x0123: Vector store to L2. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVXST_L2, 0x0123)


  // 0x0124: L2 cacheable miss from HVX store; 128B line addr not in
  // tag/coalesce buffer. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVXST_L2_MISS, 0x0124)


  // 0x0125: TCM store access for HVX (HVX store to L2 TCM space). (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVXST_L2TCM, 0x0125)


  // 0x0126: Vector store to VTCM. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVXST_VTCM, 0x0126)


  // 0x0127: L2 cacheable secondary miss from HVX store; 128B line addr not in
  // tag/coalesce buffer. (Maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVXST_L2_SECODARY_MISS, 0x0127)


  // 0x0128: Executed simple ALU instruction. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVXPIPE_ALU, 0x0128)


  // 0x0129: Executed multiply or abs-diff instruction. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVXPIPE_MPY, 0x0129)


  // 0x012A: Executed bit shift or bit count instruction. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVXPIPE_SHIFT, 0x012A)


  // 0x012B: Executed permute or cross-lane data movement instruction. (Not
  // maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVXPIPE_PERM, 0x012B)


  // 0x012C: Cycles 4 HVX contexts running concurrently. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CYCLES_4_HVX_CONTEXTS_RUNNING, 0x012C)


  // 0x0190: Cycles a thread had empty VFIFO. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_HVX_VFIFO_EMPTY, 0x0190)


  // 0x0191: Cycles 5 HVX contexts running concurrently. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CYCLES_5_HVX_CONTEXTS_RUNNING, 0x0191)


  // 0x0192: Cycles 6 HVX contexts running concurrently. (Not maskable)
HEXAGON_PMU_EVENT_DEF(HEX_PMU_EVENT_CYCLES_6_HVX_CONTEXTS_RUNNING, 0x0192)
