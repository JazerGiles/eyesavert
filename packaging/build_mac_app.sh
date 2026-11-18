#!/usr/bin/env bash
# Build, sign, notarize and zip Eyes.app for sharing (Apple Silicon only).
#
# One-time setup:
#   eyeVenv/bin/pip install pyinstaller
#   xcrun notarytool store-credentials eyes-notary \
#       --apple-id <your Apple ID email> --team-id 3J73345UK5
#     (it prompts for an app-specific password from appleid.apple.com)
#
# Usage (from anywhere):
#   packaging/build_mac_app.sh              # build + sign + notarize
#   SKIP_NOTARIZE=1 packaging/build_mac_app.sh   # build + sign only (local testing)
#
# Output: dist/Eyes.zip -- send a link to that (too big for email attachments).
set -euo pipefail

cd "$(dirname "$0")/.."

IDENTITY="${IDENTITY:-Developer ID Application: JAZER GILES SIBLEY-SCHWARTZ (3J73345UK5)}"
NOTARY_PROFILE="${NOTARY_PROFILE:-eyes-notary}"
PY="${PY:-eyeVenv/bin/python}"
APP="dist/Eyes.app"
ZIP="dist/Eyes.zip"

for f in face_landmarker.task pose_landmarker.task; do
    [[ -f "$f" ]] || { echo "missing $f -- see eye_rig.py docstring for download URLs"; exit 1; }
done

echo "==> Building"
rm -rf build dist
"$PY" -m PyInstaller --noconfirm --clean packaging/Eyes.spec

echo "==> Signing"
# --deep re-signs every nested binary with our identity + hardened runtime,
# which notarization requires of every Mach-O in the bundle.
codesign --force --deep --timestamp --options runtime \
    --entitlements packaging/entitlements.plist \
    --sign "$IDENTITY" "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"

ditto -c -k --keepParent "$APP" "$ZIP"

if [[ "${SKIP_NOTARIZE:-0}" == "1" ]]; then
    echo "==> Skipping notarization. Signed (not notarized) app: $APP"
    exit 0
fi

echo "==> Notarizing (usually a few minutes)"
xcrun notarytool submit "$ZIP" --keychain-profile "$NOTARY_PROFILE" --wait

echo "==> Stapling"
xcrun stapler staple "$APP"
rm "$ZIP"
ditto -c -k --keepParent "$APP" "$ZIP"
spctl --assess --type execute --verbose "$APP"

echo "==> Done: $ZIP ($(du -h "$ZIP" | cut -f1))"
