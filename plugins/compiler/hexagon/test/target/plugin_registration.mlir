// This test just checks that loading the plugin exposes its expected
// pass-pipeline and option registrations in the compiler CLI.
// RUN: iree-opt --help | FileCheck %s
// CHECK: --iree-hexagon-configuration-pipeline
// CHECK: --iree-hexagon-features=
// CHECK: --iree-hexagon-v=
