#! /bin/sh

set -eux

DIR="/data/local/tmp"
MAX_USAGE_PERCENT=50
CLEAN_START_AGE_DAYS=30
CLEAN_END_AGE_DAYS=7

# while SD card partition is not empty enough, delete oldest files
MAXDAYS=$CLEAN_START_AGE_DAYS
while [[ $MAXDAYS -ge $CLEAN_END_AGE_DAYS ]]
do
  USAGE_PERCENT=$(df "$DIR" | sed -n 's/^.* \([0-9]*\)%.*$/\1/p' | tail -n 1)
  if [[ $USAGE_PERCENT == "" ]]
  then
    echo "could not get disk usage" >&2
    exit 1
  fi
  if [[ $USAGE_PERCENT -le $MAX_USAGE_PERCENT ]]
  then
    echo "clean enough" >&2
    exit 0
  fi
  # remove files older than MAXDAYS and then all dirs that are empty
  find /data/local/tmp/ -mtime +$MAXDAYS -type f -delete
  find /data/local/tmp/ -type d -empty -delete
  MAXDAYS=$((MAXDAYS-1))
  if [[ $MAXDAYS -lt $CLEAN_END_AGE_DAYS ]]
  then
    echo "could not free enough space" >&2
    exit 1
  fi
done
