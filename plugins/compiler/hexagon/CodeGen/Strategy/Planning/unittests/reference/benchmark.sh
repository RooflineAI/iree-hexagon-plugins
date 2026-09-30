#!/bin/bash

DIR=$1
EXE_NAME=${2:-exe.vmfb}
LOG_NAME=${3:-adb-logcat.log}

EXE=$DIR/$EXE_NAME
INPUTS=$(cat $DIR/inputs)
FUNCTION=$(cat $DIR/function)


LOG_FILE=$DIR/$LOG_NAME

REMOTE_DIR=/data/local/tmp/$USER/


# i will manually copy the runtime, before any experiments
# so this is only for reference
#
# bazel build //plugins/runtime/hexagon:hexagon_runtime_aarch64_android
# adb push bazel-bin/plugins/runtime/hexagon/hexagon_runtime_aarch64_android.zip $REMOTE_DIR
# adb shell "unzip -o '$REMOTE_DIR/hexagon_runtime_aarch64_android.zip' -d '$REMOTE_DIR'"
# adb shell "chmod +x '$REMOTE_DIR/bin/iree-run-module'"
# adb shell "chmod +x '$REMOTE_DIR/bin/iree-benchmark-module'"

# no debug
adb shell "rm -f $REMOTE_DIR/lib/hexagon/iree-run-module.farf"

adb push $EXE $REMOTE_DIR/$EXE_NAME

adb shell "export DSP_LIBRARY_PATH=$REMOTE_DIR/lib/hexagon && \
  $REMOTE_DIR/bin/iree-benchmark-module \
    --module=$REMOTE_DIR/$EXE_NAME \
    --function=$FUNCTION \
    $INPUTS \
    --device=hexagon \
    ${BENCH_ARGS:-} \
    --output=@$REMOTE_DIR/$EXE_NAME-output.npy"
