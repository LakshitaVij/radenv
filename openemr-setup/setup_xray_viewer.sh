#!/usr/bin/env bash
# setup_xray_viewer.sh
#
# One-shot setup for the X-ray Viewer OpenEMR form on a fresh clone of this
# repo (e.g. the chartr-gyms environment) - applies the two patches that add
# the form's PHP source and its "Open" button label, seeds the DB so it's
# pre-attached to every encounter with real imaging, then builds the React
# frontend and deploys it into place. Safe to re-run: patch application is
# skipped if already applied, and the build/copy step just overwrites.
#
# Assumes: the OpenEMR docker compose stack (openemr/docker/development-easy)
# is already up, and `python3`/`npm` resolve to working interpreters - if
# python3 is broken or points at the wrong version on your machine, edit the
# PYTHON variable below before running.

set -euo pipefail

PYTHON="${PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENEMR_DIR="$REPO_ROOT/openemr"
SETUP_DIR="$REPO_ROOT/openemr-setup"
FRONTEND_DIR="$REPO_ROOT/xray-viewer-frontend"
DIST_TARGET="$OPENEMR_DIR/interface/forms/xray_viewer/public/dist"

echo "==> Applying xray_viewer_form.patch (new form source)"
cd "$OPENEMR_DIR"
if git apply --check "$SETUP_DIR/xray_viewer_form.patch" 2>/dev/null; then
  git apply "$SETUP_DIR/xray_viewer_form.patch"
  echo "    applied."
elif [ -f "interface/forms/xray_viewer/view.php" ]; then
  echo "    already applied, skipping."
else
  echo "    ERROR: patch doesn't apply and the form isn't already present - check for upstream drift." >&2
  exit 1
fi

echo "==> Applying xray-viewer-open-label.patch (forms.php button label)"
if git apply --check "$SETUP_DIR/xray-viewer-open-label.patch" 2>/dev/null; then
  git apply "$SETUP_DIR/xray-viewer-open-label.patch"
  echo "    applied."
elif git apply --reverse --check "$SETUP_DIR/xray-viewer-open-label.patch" 2>/dev/null; then
  echo "    already applied, skipping."
else
  echo "    ERROR: patch doesn't apply cleanly - forms.php may have diverged from what it was written against." >&2
  exit 1
fi

echo "==> Seeding X-ray Viewer form attachments in the DB"
cd "$SETUP_DIR"
"$PYTHON" seed_xray_viewer_forms.py

echo "==> Building the X-ray Viewer frontend"
cd "$FRONTEND_DIR"
if [ ! -d node_modules ]; then
  npm install
fi
npm run build

echo "==> Deploying build output to $DIST_TARGET"
rm -rf "$DIST_TARGET"
cp -r dist "$DIST_TARGET"

echo "==> Done. X-ray Viewer form is ready."
