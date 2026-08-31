// Copyright 2026 RooflineAI GmbH
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/* This is a helper tool to wrap execution of a process (or process tree) in a
   new session and make sure the session is terminated when any of the following
   happens:
   1) the wrapper tool is killed, even when killed with SIGKILL (e.g. as done
      by Python subprocess when hitting timeout)
   2) the parent of the wrapper tool (i.e. the process that called the wrapper)
      terminates

   Note: This will not work if a child process of the wrapped process (or any
   descendant process) calls setsid() itself to start a new session to e.g.
   spawn a daemon process. This tool is not meant as a full sandbox against
   escaping processes, but a light-weight clean-up helper.

   Detailed Description
   --------------------

   The detection of (1) or (2) is achieved by running two (see note below for
   reason) wrapper processes around the new session:
     - caller (the process executing the wrapper)
        - outer (first wrapper process)
           - middle (second wrapper process)
              - payload (actual process to be run, leader of new session)
                 - child 1 (optional)
                    - grandchild (optional)
                 - child 2 (optional)
                    - ...
                 - ...

   On startup, "outer" will start "middle" in a new session. "middle" will
   start "payload" in a new session.
   On regular termination of their child, "outer" and "middle" exit and forward
   the exit code of their child. However, "middle" will send a SIGKILL to the
   session of "payload" before exiting to make sure no process is left behind.

   In order to react to the events (1) and (2) described above, "outer" and
   "middle" record the PID of their parent on startup and periodically (every
   100ms) check if the PID of the parent is still the same.
   If not, "outer" just terminates with the exit code for SIGKILL in order to
   make "middle" detect a change of its parent. When "middle" detects a change
   of its parent, it will send SIGKILL to the session of "payload" and exit
   with the exit code SIGKILL.

   Notes:
     - Two layers of wrapping are required to enable cleanup in case (1). If
       there was only a single wrapper layer, then killing the wrapper tool
       (with SIGKILL) would leave only the payload process (and its
       subprocesses) behind. There would not be anything of the wrapper tool
       left that could kill the payload process/session.
     - The middle process needs to run in its own session to support case (1)
       if the wrapper tool is killed with a session kill (using SIGKILL). If
       it was not running in its own session, the session kill would kill
       outer and middle at the same time, leaving only the payload behind and
       no process of the wrapper is around any more to kill the playload.
   */

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define EXIT_ARGS_FAILURE (2)
#define EXIT_EXEC_FAILURE (127)
#define EXIT_SIGKILL (128 + SIGKILL)
#define EXIT_PARENT_CHANGE (EXIT_SIGKILL)

static const struct timespec poll_interval = {
    .tv_sec = 0, .tv_nsec = 100000000UL /* 100ms */};

/* Convert a wait status to an exit code. */
static int exit_code_from_status(int status) {
  if (WIFEXITED(status))
    return WEXITSTATUS(status);
  if (WIFSIGNALED(status))
    return 128 + WTERMSIG(status);
  return 1;
}

/* Wait for a child process, polling every 100ms and checking that our parent
   PID has not changed. Returns the child's exit code, 1 on error from waitpid
   and EXIT_SIGKILL when parent changes. */
static int wait_and_monitor(pid_t child, pid_t original_parent) {
  for (;;) {
    if (getppid() != original_parent) {
      return EXIT_PARENT_CHANGE;
    }

    int status;
    pid_t ret = waitpid(child, &status, WNOHANG);
    if (ret < 0) {
      if (errno == EINTR) {
        continue;
      }
      fprintf(stderr, "limit_lifetime: waitpid: %s\n", strerror(errno));
      return EXIT_FAILURE;
    }
    if (ret > 0) {
      return exit_code_from_status(status);
    }

    nanosleep(&poll_interval, NULL);
  }
}

/* "middle" process: starts payload in a new session, monitors parent (outer),
   and kills the session on exit. */
static int middle_main(char **argv) {
  pid_t original_parent_of_middle = getppid();

  /* Make middle a session leader, so it can still survive the session of outer
   * being killed (this would kill middle if no session leader) to detect the
   * killed outer and clean up the payload. */
  if (setsid() < 0) {
    fprintf(stderr, "limit_lifetime: setsid: %s\n", strerror(errno));
    return EXIT_FAILURE;
  }

  pid_t payload = fork();
  if (payload < 0) {
    fprintf(stderr, "limit_lifetime: fork (payload): %s\n", strerror(errno));
    return EXIT_FAILURE;
  }

  if (payload == 0) {
    /* Payload: create a new session and exec. */
    if (setsid() < 0) {
      fprintf(stderr, "limit_lifetime: setsid: %s\n", strerror(errno));
      _exit(EXIT_FAILURE);
    }
    execvp(argv[0], argv);
    fprintf(stderr, "limit_lifetime: exec %s: %s\n", argv[0], strerror(errno));
    _exit(EXIT_EXEC_FAILURE);
  }

  /* Middle: wait for payload. */
  int child_exit = wait_and_monitor(payload, original_parent_of_middle);

  /* Kill the entire session (negative PID = process group / session). */
  if (kill(-payload, SIGKILL) < 0 && errno == ESRCH) {
    /* The session did not exist, the payload process might not yet have
       executed setsid(). So kill the payload process explicitly. */
    kill(payload, SIGKILL);
    /* To completely avoid a race here, kill the entire session again. This is
       needed if the payload executed setsid() and execve() and a fork() just
       in between kill(-payload) and kill(payload). */
    kill(-payload, SIGKILL);
  }
  /* Note: There is no protection against payload or one of its descendants
    having executed a setsid(). This is by choice. This is not the use-case for
    this tool. */

  return child_exit;
}

/* "outer" process (the initial process): starts middle, monitors parent
   (caller). */
int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <command> [args...]\n", argv[0]);
    return EXIT_ARGS_FAILURE;
  }

  pid_t original_parent_of_outer = getppid();

  pid_t middle = fork();
  if (middle < 0) {
    fprintf(stderr, "limit_lifetime: fork (middle): %s\n", strerror(errno));
    return EXIT_FAILURE;
  }

  if (middle == 0) {
    /* Middle process. */
    _exit(middle_main(argv + 1));
  }

  /* Outer: wait for middle, checking parent. */
  return wait_and_monitor(middle, original_parent_of_outer);
}
