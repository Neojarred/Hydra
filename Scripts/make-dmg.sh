#!/bin/bash
# Builds Hydra.dmg from the application bundle.
#
# A disk image is the expected format on macOS: it mounts with a double-click, and the
# shortcut to /Applications says without words what to do. A zip archive
# works too but leaves the user to decide where to put the app, which often ends up in the
# Downloads folder.
#
# What this script does NOT do: notarize. Without an Apple developer account the app stays
# ad-hoc signed, and macOS will refuse to open it on the first try. See README.md.
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIGURATION="${1:-release}"
APP=".build/${CONFIGURATION}/Hydra.app"
VERSION=$(plutil -extract CFBundleShortVersionString raw "${APP}/Contents/Info.plist" 2>/dev/null || echo "0.0.0")
DMG=".build/Hydra-${VERSION}.dmg"

if [ ! -d "${APP}" ]; then
  echo "✘ ${APP} not found, run Scripts/build-app.sh first" >&2
  exit 1
fi

echo "→ checking the signature"
codesign --verify --deep --strict "${APP}" && echo "  signature valid"

# The contents are assembled in a temporary folder: hdiutil copies whatever it finds there,
# and a clean folder avoids shipping working files.
STAGING=$(mktemp -d)
trap 'rm -rf "${STAGING}"' EXIT

echo "→ assembling"
cp -R "${APP}" "${STAGING}/"
ln -s /Applications "${STAGING}/Applications"

echo "→ compression"
rm -f "${DMG}"
hdiutil create \
  -volname "Hydra ${VERSION}" \
  -srcfolder "${STAGING}" \
  -ov -format UDZO \
  "${DMG}" >/dev/null

echo "→ ${DMG}"
du -h "${DMG}" | awk '{print "  " $1}'
shasum -a 256 "${DMG}" | awk '{print "  sha256 " $1}'
