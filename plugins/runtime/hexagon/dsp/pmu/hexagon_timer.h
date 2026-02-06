#ifndef HEXAGON_TIMER_H
#define HEXAGON_TIMER_H

#include "hexagon/arm_dsp/profiling.h"
#include <stdint.h>

static inline uint64_t read_timer() {
  uint64_t ret = 0;
  __asm__ __volatile__(" %0 = UTIMER \n" : "=r"(ret));
  return ret;
}

static inline double timer_measurement_convert_to_us(uint64_t v) {
  double us = (double)v;
  us /= tick_timer_freq_MHz;
  return us;
}

typedef struct {
  uint64_t start;
  uint64_t stop;
  uint64_t elapsed;
} measurement_t;

static inline void timing_measurement_start(measurement_t *utimer) {
  utimer->start = read_timer();
}

static inline void timing_measurement_stop(measurement_t *utimer) {
  utimer->stop = read_timer();
  utimer->elapsed = utimer->stop - utimer->start;
}

static inline double timing_measurement_get_start_us(measurement_t *utimer) {
  return timer_measurement_convert_to_us(utimer->start);
}

static inline double timing_measurement_get_stop_us(measurement_t *utimer) {
  return timer_measurement_convert_to_us(utimer->stop);
}

#endif // HEXAGON_TIMER_H
