#! /bin/bash

# Applies the patches under patches/<submodule>/ to the corresponding
# third-party/<submodule> checkout. Bazel reads those checkouts directly (via
# local_path_override / new_local_repository), so these patches need to be
# applied to their working trees - not just recorded as patch files - for the
# build to work.
#
# Safe to re-run any time (e.g. after `git submodule update` resets a
# checkout back to its pinned commit and wipes these edits): already-applied
# patches are skipped.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
patches_dir="${repo_root}/patches"

# One entry per patched submodule: "<patches subdir name>:<submodule path>".
submodules=(
    "iree:third-party/iree"
    "hexagon-mlir:third-party/hexagon-mlir"
)

for entry in "${submodules[@]}"; do
    patch_subdir="${entry%%:*}"
    submodule_relpath="${entry#*:}"
    submodule_dir="${repo_root}/${submodule_relpath}"

    for patch in "${patches_dir}/${patch_subdir}"/*.patch; do
        name="$(basename "${patch}")"
        if git -C "${submodule_dir}" apply -p1 --reverse --check "${patch}" 2>/dev/null; then
            echo "skip (already applied): ${submodule_relpath}: ${name}"
        elif git -C "${submodule_dir}" apply -p1 --check "${patch}" 2>/dev/null; then
            git -C "${submodule_dir}" apply -p1 "${patch}"
            echo "applied: ${submodule_relpath}: ${name}"
        else
            echo "ERROR: ${name} does not apply cleanly to ${submodule_relpath} - it may need to be rebased onto the current pinned commit." >&2
            exit 1
        fi
    done
done
