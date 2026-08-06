#!/bin/bash
# Assemble Hydra.app à partir de l'exécutable SwiftPM.
#
# SwiftPM ne sait pas produire de paquet applicatif : il compile un exécutable et, à
# côté, les paquets de ressources des cibles qui en déclarent. Une application SwiftUI a
# besoin des deux réunis dans une arborescence .app avec son Info.plist — sans quoi elle
# n'a ni icône dans le Dock, ni barre de menus, et les ressources restent introuvables.
#
# Ce script fait ce travail sans dépendre d'un projet Xcode, ce qui garde la construction
# reproductible depuis les sources (D-016).
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIGURATION="${1:-release}"
BUILD_DIR=".build/${CONFIGURATION}"
APP="${BUILD_DIR}/Hydra.app"
VERSION="0.1.0"

echo "→ compilation (${CONFIGURATION})"
swift build -c "${CONFIGURATION}" --product HydraApp

echo "→ assemblage du paquet"
rm -rf "${APP}"
mkdir -p "${APP}/Contents/MacOS" "${APP}/Contents/Resources"

cp "${BUILD_DIR}/HydraApp" "${APP}/Contents/MacOS/Hydra"

# Les paquets de ressources SwiftPM — dont les sources Metal, compilées à l'exécution —
# vont dans Contents/Resources.
#
# C'est l'emplacement conventionnel, et `Bundle.module` l'inspecte en premier. Les placer
# à côté de l'exécutable fonctionne aussi, mais `codesign` refuse alors de signer le
# paquet : il ne s'attend pas à trouver un bundle imbriqué dans Contents/MacOS.
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

# Signature ad hoc : suffisante pour lancer localement et pour une distribution par
# sources. Une notarisation demanderait un compte développeur Apple.
# Signature du seul paquet applicatif : les paquets de ressources SwiftPM n'ont pas la
# structure qu'attend codesign, et n'ont pas besoin d'être signés séparément.
if codesign --force --sign - "${APP}" 2>/dev/null; then
  echo "  signée ad hoc"
else
  echo "  (signature ad hoc impossible — l'application reste lançable)"
fi

echo "→ autotest"
"${APP}/Contents/MacOS/Hydra" --self-test

echo "→ ${APP}"
du -sh "${APP}" | awk '{print "  " $1}'
