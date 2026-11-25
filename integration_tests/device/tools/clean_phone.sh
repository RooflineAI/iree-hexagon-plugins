#! /bin/sh

set -eux

adb push "$(dirname "$0")/phone/clean_data_local_tmp.sh" /data/local/tmp/
adb shell /data/local/tmp/clean_data_local_tmp.sh
