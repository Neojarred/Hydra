#!/bin/bash
# Runs the test suite.
#
# This machine has only the Command Line Tools, not Xcode: swift-testing is present but not
# on SwiftPM's default search paths. We add them explicitly. With Xcode installed, a plain
# `swift test` would do.
set -euo pipefail

CLT="/Library/Developer/CommandLineTools"
FRAMEWORKS="$CLT/Library/Developer/Frameworks"
INTEROP="$CLT/Library/Developer/usr/lib"

if [ ! -d "$FRAMEWORKS/Testing.framework" ]; then
  echo "Testing.framework not found under $FRAMEWORKS" >&2
  exit 1
fi

exec swift test \
  -Xswiftc -F -Xswiftc "$FRAMEWORKS" \
  -Xlinker -rpath -Xlinker "$FRAMEWORKS" \
  -Xlinker -rpath -Xlinker "$INTEROP" \
  "$@"
