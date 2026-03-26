// This test just checks that loading the plugin exposes its expected
// pass-pipeline and option registrations in the compiler CLI.
// RUN: iree-opt --iree-load-plugin=cellar_hexagon=$CELLAR_HEXAGON_COMPILER_PLUGIN --help | FileCheck %s
// CHECK: --iree-hexagon-configuration-pipeline
// CHECK: --iree-hexagon-features=
// CHECK: --iree-hexagon-v=
