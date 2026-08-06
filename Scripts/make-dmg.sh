#!/bin/bash
# Fabrique Hydra.dmg à partir du paquet applicatif.
#
# Une image disque est le format attendu sur macOS : elle se monte d'un double-clic, et le
# raccourci vers /Applications indique sans texte ce qu'il faut faire. Une archive zip
# fonctionne aussi mais laisse l'utilisateur décider où poser l'application, ce qui finit
# souvent dans le dossier Téléchargements.
#
# Ce que ce script ne fait PAS : notariser. Sans compte développeur Apple, l'application
# reste signée ad hoc, et macOS refusera de l'ouvrir au premier essai. Voir README.md.
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIGURATION="${1:-release}"
APP=".build/${CONFIGURATION}/Hydra.app"
VERSION=$(plutil -extract CFBundleShortVersionString raw "${APP}/Contents/Info.plist" 2>/dev/null || echo "0.0.0")
DMG=".build/Hydra-${VERSION}.dmg"

if [ ! -d "${APP}" ]; then
  echo "✘ ${APP} introuvable — lancez d'abord Scripts/build-app.sh" >&2
  exit 1
fi

echo "→ vérification de la signature"
codesign --verify --deep --strict "${APP}" && echo "  signature valide"

# Le contenu est assemblé dans un dossier temporaire : hdiutil copie ce qu'il y trouve, et
# un dossier propre évite d'embarquer des fichiers de travail.
STAGING=$(mktemp -d)
trap 'rm -rf "${STAGING}"' EXIT

echo "→ assemblage"
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
