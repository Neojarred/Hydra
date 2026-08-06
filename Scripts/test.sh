#!/bin/bash
# Lance la suite de tests.
#
# Cette machine n'a que les Command Line Tools, pas Xcode : swift-testing y est présent
# mais n'est pas sur les chemins de recherche par défaut de SwiftPM. On les ajoute
# explicitement. Avec Xcode installé, un simple `swift test` suffirait.
set -euo pipefail

CLT="/Library/Developer/CommandLineTools"
FRAMEWORKS="$CLT/Library/Developer/Frameworks"
INTEROP="$CLT/Library/Developer/usr/lib"

if [ ! -d "$FRAMEWORKS/Testing.framework" ]; then
  echo "Testing.framework introuvable sous $FRAMEWORKS" >&2
  exit 1
fi

exec swift test \
  -Xswiftc -F -Xswiftc "$FRAMEWORKS" \
  -Xlinker -rpath -Xlinker "$FRAMEWORKS" \
  -Xlinker -rpath -Xlinker "$INTEROP" \
  "$@"
