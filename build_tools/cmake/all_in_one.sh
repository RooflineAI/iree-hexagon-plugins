#! /bin/bash

set -eux -o pipefail

SCRIPT_DIR=$(dirname "$0")

if ! which cmake || ! which ninja
then
  sudo apt-get update
  sudo apt-get install -y cmake ninja-build
fi

if [[ ! -r /usr/lib/llvm-19/bin/clang ]]
then
  wget -O /tmp/llvm.sh https://apt.llvm.org/llvm.sh
  chmod +x /tmp/llvm.sh
  sudo /tmp/llvm.sh 19
fi

if ! which unzip
then
  sudo apt-get update
  sudo apt-get install -y unzip
fi

NDK_VERSION=r28c
NDK_FOLDER=28.2.13676358
export ANDROID_NDK_HOME="/opt/android-sdk/ndk/${NDK_FOLDER}"
if [[ ! -d $ANDROID_NDK_HOME ]]
then
  NDK_ZIP="android-ndk-${NDK_VERSION}-linux.zip"
  NDK_SHA256="dfb20d396df28ca02a8c708314b814a4d961dc9074f9a161932746f815aa552f"
  wget https://dl.google.com/android/repository/${NDK_ZIP} -O /tmp/ndk.zip
  echo "${NDK_SHA256} /tmp/ndk.zip" | sha256sum -c -
  sudo unzip -q /tmp/ndk.zip -d /opt
  sudo mkdir -p "$(dirname "$ANDROID_NDK_HOME")"
  sudo mv "/opt/android-ndk-${NDK_VERSION}" "$ANDROID_NDK_HOME"
fi

"$SCRIPT_DIR/../update_submodules.py"

"$SCRIPT_DIR/build_and_package.sh"

echo "all done"
