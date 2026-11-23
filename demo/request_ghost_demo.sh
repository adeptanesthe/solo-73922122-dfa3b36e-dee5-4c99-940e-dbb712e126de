#!/usr/bin/env bash
# Afterimage verification walkthrough against a running service.
#
#   1. publish a small two-tone page and note the current version;
#   2. upload the previous-round PNG plus ONE screen photograph: the only
#      feasible residual-ratio interval straddles 20%, so the verdict is
#      "reshoot" (补拍);
#   3. add a SECOND photograph of the same screen with a different exposure:
#      their common interval is fully below 20%, so the verdict changes to
#      "no_reflash" (无需重刷);
#   4. as a contrast, send one strongly ghosted photograph and show
#      "reflash" (建议重刷);
#   5. send two mutually incompatible photographs and show the 409 evidence
#      conflict; then call a stale version and show the 409 version conflict.
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ASSETS="${ASSETS:-$HERE/out/ghost}"
mkdir -p "$ASSETS"
python3 "$HERE/make_ghost_demo.py" "$ASSETS" >/dev/null

echo "== publish the current page as version 1 =="
CREATE=$(curl -fsS -X POST "$BASE/api/packages" \
  -F "rgb_pages=@$ASSETS/current_rgb.png" \
  -F "label_pages=@$ASSETS/current_labels.png" \
  -F 'palette=["#000000","#ffffff"]' \
  -F "min_distance=60")
echo "$CREATE" | jq '{package_id, version}'
PACKAGE=$(echo "$CREATE" | jq -r .package_id)
VERSION=$(echo "$CREATE" | jq -r .version)
ENDPOINT="$BASE/api/packages/$PACKAGE/versions/$VERSION/pages/1/ghost-check"

echo
echo "== 1) one photograph: residual interval straddles 20% -> reshoot =="
curl -fsS -X POST "$ENDPOINT" \
  -F "old_image=@$ASSETS/previous_delivery.png" \
  -F "photos=@$ASSETS/shot_ambiguous.png" \
  | jq '{decision, residual_ratio, calibration_points: (.calibration_points|length),
         region_of_interest, photo: .photos[0] | {gain_a, offset_b, residual_ratio_interval}}'

echo
echo "== 2) add a second photograph: common interval below 20% -> no_reflash =="
curl -fsS -X POST "$ENDPOINT" \
  -F "old_image=@$ASSETS/previous_delivery.png" \
  -F "photos=@$ASSETS/shot_ambiguous.png" \
  -F "photos=@$ASSETS/shot_clear_a.png" \
  | jq '{decision, residual_ratio,
         photos: [.photos[] | {photo, gain_a, offset_b, residual_ratio_interval}]}'

echo
echo "== 3) strong ghost on one photograph -> reflash =="
curl -fsS -X POST "$ENDPOINT" \
  -F "old_image=@$ASSETS/previous_delivery.png" \
  -F "photos=@$ASSETS/shot_high_ghost.png" \
  | jq '{decision, residual_ratio, photo: .photos[0]}'

echo
echo "== 4) two incompatible photographs -> 409 evidence_conflict =="
CONFLICT=$(curl -sS -w '\n%{http_code}' -X POST "$ENDPOINT" \
  -F "old_image=@$ASSETS/previous_delivery.png" \
  -F "photos=@$ASSETS/shot_clear_a.png" \
  -F "photos=@$ASSETS/shot_high_ghost.png")
echo "HTTP $(echo "$CONFLICT" | tail -n1)"
echo "$CONFLICT" | sed '$d' | jq '.error | {code, message, photo_intervals}'

echo
echo "== 5) bump the package version, then call the stale endpoint -> 409 =="
curl -fsS -X POST "$BASE/api/packages/$PACKAGE/locks" \
  -H 'content-type: application/json' \
  -d '{"label": 1, "colour": "#000000"}' | jq '{version}'
STALE=$(curl -sS -w '\n%{http_code}' -X POST \
  "$BASE/api/packages/$PACKAGE/versions/$VERSION/pages/1/ghost-check" \
  -F "old_image=@$ASSETS/previous_delivery.png" \
  -F "photos=@$ASSETS/shot_ambiguous.png")
echo "HTTP $(echo "$STALE" | tail -n1)"
echo "$STALE" | sed '$d' | jq '.error | {code, message, current_version}'
