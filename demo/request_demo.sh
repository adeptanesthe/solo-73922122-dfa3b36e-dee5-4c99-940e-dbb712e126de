#!/usr/bin/env bash
# End-to-end walkthrough against a running service (default http://localhost:8000).
#
#   1. create a package from the generated demo pages;
#   2. lock the seal to the same blue as the sky;
#   3. show that the rule label is forced to change colour on *every* page
#      because label colours are shared across the set;
#   4. request an impossible lock and show the 409 + preserved version;
#   5. download a superseded version and show the 409.
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PAGES="$HERE/pages"
WORK="${WORK:-$HERE/out}"
mkdir -p "$WORK"

if [[ ! -f "$PAGES/manifest.json" ]]; then
  python3 "$HERE/make_demo_pages.py" "$PAGES" >/dev/null
fi

PALETTE='["#101010","#f2eee2","#a12f26","#2f5d8a","#d8b41e"]'
DISTANCE=200

echo "== create package =="
CREATE=$(curl -fsS -X POST "$BASE/api/packages" \
  -F "rgb_pages=@$PAGES/page1_rgb.png" \
  -F "label_pages=@$PAGES/page1_labels.png" \
  -F "rgb_pages=@$PAGES/page2_rgb.png" \
  -F "label_pages=@$PAGES/page2_labels.png" \
  -F "rgb_pages=@$PAGES/page3_rgb.png" \
  -F "label_pages=@$PAGES/page3_labels.png" \
  -F "palette=$PALETTE" \
  -F "min_distance=$DISTANCE")
echo "$CREATE" | jq '{package_id, version, total_error, locks,
  legend: [.entries[] | {label: .label, colour: .colour, pages: .pages}]}'

PACKAGE=$(echo "$CREATE" | jq -r .package_id)
V1=$(echo "$CREATE" | jq -r .version)

echo
echo "== legend before the lock (label 160 = seal, 96 = rule, 32 = sky) =="
BEFORE=$(echo "$CREATE" | jq -c '[.entries[] | {(.label|tostring): .colour}] | add')
echo "$BEFORE" | jq .

echo
echo "== lock label 160 (seal) to the sky blue #2f5d8a =="
LOCKED=$(curl -fsS -X POST "$BASE/api/packages/$PACKAGE/locks" \
  -H 'content-type: application/json' \
  -d '{"label": 160, "colour": "#2f5d8a"}')
echo "$LOCKED" | jq '{version, total_error, change: .change, locks,
  legend: [.entries[] | {label: .label, colour: .colour, pages: .pages, locked: .locked}]}'
V2=$(echo "$LOCKED" | jq -r .version)
AFTER=$(echo "$LOCKED" | jq -c '[.entries[] | {(.label|tostring): .colour}] | add')
echo
echo "colours that changed because of the one lock:"
jq -n --argjson a "$BEFORE" --argjson b "$AFTER" \
  '$a | to_entries | map(select(.value != $b[.key])) | map({label: .key, before: .value, after: $b[.key]})'

echo
echo "== download version $V2 (all pages + legend) =="
curl -fsS -o "$WORK/package-v$V2.zip" \
  "$BASE/api/packages/$PACKAGE/versions/$V2/download"
unzip -l "$WORK/package-v$V2.zip" | sed -n '1,12p'

echo
echo "== try an impossible lock: label 96 (rule) also blue while touching the seal =="
CONFLICT=$(curl -sS -w '\n%{http_code}' -X POST "$BASE/api/packages/$PACKAGE/locks" \
  -H 'content-type: application/json' \
  -d '{"label": 96, "colour": "#2f5d8a"}')
CODE=$(echo "$CONFLICT" | tail -n1)
BODY=$(echo "$CONFLICT" | sed '$d')
echo "HTTP $CODE"
echo "$BODY" | jq '.error | {code, message, kept_version, kept_locks,
  conflict_type: .details.type, labels: .details.labels,
  colour_pairs: .details.colour_pairs, locks_cause: .details.locks_cause}'

echo
echo "== current package was preserved after the conflict =="
curl -fsS "$BASE/api/packages/$PACKAGE" | jq '{version, locks}'

echo
echo "== superseded version $V1 now conflicts =="
STALE=$(curl -sS -w '\n%{http_code}' \
  "$BASE/api/packages/$PACKAGE/versions/$V1/download")
echo "HTTP $(echo "$STALE" | tail -n1)"
echo "$STALE" | sed '$d' | jq '.error'

echo
echo "== unlock label 160 and bump to a new version =="
curl -fsS -X DELETE "$BASE/api/packages/$PACKAGE/locks/160" \
  | jq '{version, locks, change: .change}'

echo
echo "artifacts in $WORK"
