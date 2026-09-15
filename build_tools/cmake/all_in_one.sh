#! /bin/bash

set -eux -o pipefail

SCRIPT_DIR=$(dirname "$0")

packages_to_install=()

require_command() {
  local command_name="$1"
  local package_name="$2"
  if ! command -v "$command_name" >/dev/null 2>&1
  then
    packages_to_install+=("$package_name")
  fi
}

require_command cmake cmake
require_command curl curl
require_command ninja ninja-build
require_command realpath coreutils
require_command sha256sum coreutils
require_command tar tar
require_command unzip unzip
require_command xz xz-utils
require_command zip zip

if (( ${#packages_to_install[@]} > 0 ))
then
  sudo apt-get update
  sudo apt-get install -y "${packages_to_install[@]}"
fi

if [[ ! -x /usr/lib/llvm-19/bin/clang || ! -x /usr/lib/llvm-19/bin/clang++ ]]
then
  curl -fsSL https://apt.llvm.org/llvm.sh -o /tmp/llvm.sh
  chmod +x /tmp/llvm.sh
  sudo /tmp/llvm.sh 19
fi

NDK_VERSION=r28c
NDK_FOLDER=28.2.13676358
export ANDROID_NDK_HOME="/opt/android-sdk/ndk/${NDK_FOLDER}"
if [[ ! -d $ANDROID_NDK_HOME ]]
then
  NDK_ZIP="android-ndk-${NDK_VERSION}-linux.zip"
  NDK_SHA256="dfb20d396df28ca02a8c708314b814a4d961dc9074f9a161932746f815aa552f"
  curl -fsSL https://dl.google.com/android/repository/${NDK_ZIP} -o /tmp/ndk.zip
  echo "${NDK_SHA256} /tmp/ndk.zip" | sha256sum -c -
  sudo unzip -q /tmp/ndk.zip -d /opt
  sudo mkdir -p "$(dirname "$ANDROID_NDK_HOME")"
  sudo mv "/opt/android-ndk-${NDK_VERSION}" "$ANDROID_NDK_HOME"
fi

"$SCRIPT_DIR/../apply_submodule_patches.sh"

"$SCRIPT_DIR/build_and_package.sh"

echo "all done"