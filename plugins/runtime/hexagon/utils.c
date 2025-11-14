// Copyright 2025 RooflineAI GmbH

#include "hexagon/utils.h"

#include <errno.h>
#include <iree/base/status.h>

iree_status_code_t iree_hal_hexagon_status_code_from_errno(void) {
  switch (errno) {
  case ECANCELED:
    return IREE_STATUS_CANCELLED;
  case EBADF:
  case EFAULT:
  case EINVAL:
    return IREE_STATUS_INVALID_ARGUMENT;
  case ENODEV:
  case ENOTDIR:
  case ENOENT:
  case ENXIO:
    return IREE_STATUS_NOT_FOUND;
  case EEXIST:
    return IREE_STATUS_ALREADY_EXISTS;
  case EACCES:
  case EPERM:
  case EROFS:
    return IREE_STATUS_PERMISSION_DENIED;
  case ENOMEM:
  case EIO:
  case EMFILE:
  case ENOSPC:
    return IREE_STATUS_RESOURCE_EXHAUSTED;
  case E2BIG:
  case EFBIG:
    return IREE_STATUS_OUT_OF_RANGE;
  case EAGAIN:
  case EBUSY:
    return IREE_STATUS_DEFERRED;
  case ENOEXEC:
    return IREE_STATUS_INCOMPATIBLE;
  }
  return IREE_STATUS_UNKNOWN;
}

iree_status_t iree_hal_hexagon_parse_bool_from_string(iree_string_view_t str,
                                                      int *parsed) {
  static const struct str_parsed_s {
    const char *str;
    int parsed;
  } accepted[] = {
      {.str = "true", .parsed = 1}, {.str = "false", .parsed = 0},
      {.str = "yes", .parsed = 1},  {.str = "no", .parsed = 0},
      {.str = "y", .parsed = 1},    {.str = "n", .parsed = 0},
      {.str = "1", .parsed = 1},    {.str = "0", .parsed = 0},
  };
  for (size_t i = 0; i < sizeof(accepted) / sizeof(accepted[0]); ++i) {
    if (iree_string_view_equal_case(str,
                                    iree_make_cstring_view(accepted[i].str))) {
      *parsed = accepted[i].parsed;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "cannot parse boolean argument");
}
