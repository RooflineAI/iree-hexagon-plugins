// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file contains a table with the event ids from the programming guide. It
// may be used to configure the pmu.

#ifndef HEXAGON_PMU_EVENTS_FROM_TABLE_H
#define HEXAGON_PMU_EVENTS_FROM_TABLE_H

#define HEXAGON_PMU_COUNTERS 8

// This is a moderately large table (+/- 34 bytes per entry and 305 entries
// -> 10kB) that is generated and used in both dsp and arm side.
// If it becomes problematic, it should be possible to move it into the arm side
// exclusively or completely removed, since it is here just for convenience

typedef enum hexagon_pmu_event_id_e {
#define HEXAGON_PMU_EVENT_DEF(name, value) name = value,
#include "hexagon_pmu_events_table.inc"
#undef HEXAGON_PMU_EVENT_DEF
} hexagon_pmu_event_id_t;

typedef struct {
  int id;
  const char *name;
} hexagon_pmu_event_desc_t;

static const hexagon_pmu_event_desc_t kHexagonPmuEventTable[] = {
#define HEXAGON_PMU_EVENT_DEF(name, value) {value, #name},
#include "hexagon_pmu_events_table.inc"
#undef HEXAGON_PMU_EVENT_DEF
};

static inline const char *hexagon_pmu_event_name(int id) {
  const unsigned int count = (unsigned int)(sizeof(kHexagonPmuEventTable) /
                                            sizeof(kHexagonPmuEventTable[0]));
  for (unsigned int i = 0; i < count; ++i) {
    if (kHexagonPmuEventTable[i].id == id) {
      return kHexagonPmuEventTable[i].name;
    }
  }
  return 0;
}

#endif // HEXAGON_PMU_EVENTS_FROM_TABLE_H
