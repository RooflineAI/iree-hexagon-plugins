// Copyright 2025 RooflineAI GmbH

#ifndef HEXAGON_PMU_H_
#define HEXAGON_PMU_H_

#include "HAP_farf.h"
#include "hexagon/arm_dsp/profiling.h"
#include "qurt_consts.h"
#include "qurt_error.h"
#include "qurt_pmu.h"
#include <stdint.h>

static inline void hexagon_pmu_read(hexagon_pmu_counters_t *counters) {
  if (qurt_pmu_get_pmucnt(counters->cts) != QURT_EOK) {
    FARF(RUNTIME_HIGH, "HEXAGON-RUNTIME-ERROR: qurt_pmu_get_pmucnt failed");
  }
}

// Configure PMU counters based on event IDs.
// See hexagon_pmu_events_ids.h for available IDs.
static inline void hexagon_pmu_configure(const hexagon_pmu_counters_ids_t *in) {
  uint32_t pmuevtcfg = 0;
  uint32_t pmuevtcfg1 = 0;
  uint32_t pmucfg = 0;

  for (int i = 0; i < 4; i++) {
    pmuevtcfg |= (in->ids[i] & 0xff) << 8 * i;
    pmucfg |= ((in->ids[i] & 0x300) >> 8) << 2 * i;
    pmuevtcfg1 |= (in->ids[i + 4] & 0xff) << 8 * i;
    pmucfg |= ((in->ids[i + 4] & 0x300) >> 8) << (2 * i + 8);
  }

  qurt_pmu_set(QURT_PMUEVTCFG, pmuevtcfg);
  qurt_pmu_set(QURT_PMUEVTCFG1, pmuevtcfg1);
  qurt_pmu_set(QURT_PMUCFG, pmucfg);
}

static inline void hexagon_pmu_log_delta(const hexagon_pmu_counters_t *before,
                                         const hexagon_pmu_counters_t *after) {
  if (!before || !after) {
    return;
  }
  FARF(RUNTIME_HIGH,
       "QURT PMU delta: c0=%u c1=%u c2=%u c3=%u c4=%u c5=%u c6=%u c7=%u",
       (unsigned)(after->cts[0] - before->cts[0]),
       (unsigned)(after->cts[1] - before->cts[1]),
       (unsigned)(after->cts[2] - before->cts[2]),
       (unsigned)(after->cts[3] - before->cts[3]),
       (unsigned)(after->cts[4] - before->cts[4]),
       (unsigned)(after->cts[5] - before->cts[5]),
       (unsigned)(after->cts[6] - before->cts[6]),
       (unsigned)(after->cts[7] - before->cts[7]));
}

static inline void hexagon_pmu_dump_registers(const char *tag) {
  const char *label = (tag && *tag) ? tag : "snapshot";
  unsigned int pmu_cnt[8] = {0};
  unsigned int pmu_cntstid[8] = {0};

  for (int i = 0; i < 4; ++i) {
    pmu_cnt[i] = qurt_pmu_get(QURT_PMUCNT0 + i);
    pmu_cntstid[i] = qurt_pmu_get(QURT_PMUCNTSTID0 + i);
    pmu_cnt[i + 4] = qurt_pmu_get(QURT_PMUCNT4 + i);
    pmu_cntstid[i + 4] = qurt_pmu_get(QURT_PMUCNTSTID4 + i);
  }

  unsigned int pmu_cfg = qurt_pmu_get(QURT_PMUCFG);
  unsigned int pmu_evtcfg = qurt_pmu_get(QURT_PMUEVTCFG);
  unsigned int pmu_evtcfg1 = qurt_pmu_get(QURT_PMUEVTCFG1);
  unsigned int pmu_stid0 = qurt_pmu_get(QURT_PMUSTID0);
  unsigned int pmu_stid1 = qurt_pmu_get(QURT_PMUSTID1);

  FARF(RUNTIME_HIGH, "QURT PMU dump (%s)", label);
  FARF(RUNTIME_HIGH, "  CFG: PMUCFG=0x%08x EVTCFG=0x%08x EVTCFG1=0x%08x",
       pmu_cfg, pmu_evtcfg, pmu_evtcfg1);
  FARF(RUNTIME_HIGH, "  STID: PMUSTID0=0x%08x PMUSTID1=0x%08x", pmu_stid0,
       pmu_stid1);
  FARF(RUNTIME_HIGH, "  CNTSTID0-3: 0x%08x 0x%08x 0x%08x 0x%08x",
       pmu_cntstid[0], pmu_cntstid[1], pmu_cntstid[2], pmu_cntstid[3]);
  FARF(RUNTIME_HIGH, "  CNTSTID4-7: 0x%08x 0x%08x 0x%08x 0x%08x",
       pmu_cntstid[4], pmu_cntstid[5], pmu_cntstid[6], pmu_cntstid[7]);
  FARF(RUNTIME_HIGH,
       "  CNT0-3: %10u (0x%08x) %10u (0x%08x) %10u (0x%08x) %10u (0x%08x)",
       pmu_cnt[0], pmu_cnt[0], pmu_cnt[1], pmu_cnt[1], pmu_cnt[2], pmu_cnt[2],
       pmu_cnt[3], pmu_cnt[3]);
  FARF(RUNTIME_HIGH,
       "  CNT4-7: %10u (0x%08x) %10u (0x%08x) %10u (0x%08x) %10u (0x%08x)",
       pmu_cnt[4], pmu_cnt[4], pmu_cnt[5], pmu_cnt[5], pmu_cnt[6], pmu_cnt[6],
       pmu_cnt[7], pmu_cnt[7]);
}

#endif // HEXAGON_PMU_H_
