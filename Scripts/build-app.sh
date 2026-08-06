#!/bin/bash
# Assembles Hydra.app from the SwiftPM executable.
#
# SwiftPM cannot produce an application bundle: it compiles an executable and, alongside it,
# the resource bundles of targets that declare any. A SwiftUI app needs both brought together
# in a .app tree with its Info.plist — without which it has no Dock icon, no menu bar, and its
# resources cannot be found.
#
# This script does that work without depending on an Xcode project, which keeps the build
# reproducible from source (D-016).
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIGURATION="${1:-release}"
BUILD_DIR=".build/${CONFIGURATION}"
APP="${BUILD_DIR}/Hydra.app"
VERSION="0.1.0"

echo "→ compiling (${CONFIGURATION})"
swift build -c "${CONFIGURATION}" --product HydraApp

echo "→ assembling the bundle"
rm -rf "${APP}"
mkdir -p "${APP}/Contents/MacOS" "${APP}/Contents/Resources"

cp "${BUILD_DIR}/HydraApp" "${APP}/Contents/MacOS/Hydra"

# The SwiftPM resource bundles — including the Metal sources, compiled at runtime — go into
# Contents/Resources.
#
# That is the conventional location, and `Bundle.module` looks there first. Putting them next
# to the executable works too, but `codesign` then refuses to sign the bundle: it does not
# expect a nested bundle inside Contents/MacOS.
for bundle in "${BUILD_DIR}"/*.bundle; do
  [ -e "${bundle}" ] || continue
  cp -R "${bundle}" "${APP}/Contents/Resources/"
done

ICON_KEY=""
if [ -f "Resources/AppIcon.icns" ]; then
  cp "Resources/AppIcon.icns" "${APP}/Contents/Resources/"
  ICON_KEY="    <key>CFBundleIconFile</key>          <string>AppIcon</string>"
fi

cat > "${APP}/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>              <string>Hydra</string>
    <key>CFBundleDisplayName</key>       <string>Hydra</string>
    <key>CFBundleExecutable</key>        <string>Hydra</string>
${ICON_KEY}
    <key>CFBundleIdentifier</key>        <string>dev.hydra.app</string>
    <key>CFBundlePackageType</key>       <string>APPL</string>
    <key>CFBundleShortVersionString</key><string>${VERSION}</string>
    <key>CFBundleVersion</key>           <string>${VERSION}</string>
    <key>LSMinimumSystemVersion</key>    <string>26.0</string>
    <key>NSHighResolutionCapable</key>   <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key><true/>
    <key>LSApplicationCategoryType</key> <string>public.app-category.developer-tools</string>
</dict>
</plist>
PLIST

# Ad-hoc signing: enough to launch locally and for source distribution. Notarization would
# require an Apple developer account.
# Only the application bundle is signed: SwiftPM's resource bundles do not have the structure
# codesign expects, and do not need signing separately.
if codesign --force --sign - "${APP}" 2>/dev/null; then
  echo "  ad-hoc signed"
else
  echo "  (ad-hoc signing failed — the app is still launchable)"
fi

echo "→ self-test"
"${APP}/Contents/MacOS/Hydra" --self-test

echo "→ ${APP}"
du -sh "${APP}" | awk '{print "  " $1}'
