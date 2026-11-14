#!/usr/bin/env bash
# Download the two public datasets the possession model is trained on, into data/external/.
#
#   bash scripts/download_public_data.sh
#
# 1. Metrica Sports sample data (3 matches, tracking + events)
#    https://github.com/metrica-sports/sample-data
# 2. IDSSE / DFL open data (7 Bundesliga matches, TRACAB tracking + events), CC-BY 4.0,
#    data owner Deutsche Fussball Liga. Bassek, Weber, Rein & Memmert (2025), "An integrated
#    dataset of spatiotemporal and event data in elite soccer", Scientific Data,
#    doi:10.1038/s41597-025-04505-y. Mirror: https://huggingface.co/datasets/pysport/idsse-data
#
# Downloads resume if interrupted (curl -C -), so re-running is safe. About 2.9 GB in total.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXTERNAL="$REPO_ROOT/data/external"
mkdir -p "$EXTERNAL/idsse"

if [ ! -d "$EXTERNAL/metrica-sample-data/.git" ]; then
  git clone --depth 1 https://github.com/metrica-sports/sample-data.git "$EXTERNAL/metrica-sample-data"
fi

IDSSE_BASE="https://huggingface.co/datasets/pysport/idsse-data/resolve/main"
MATCHES="000001_DFL-MAT-J03WMX 000001_DFL-MAT-J03WN1 000002_DFL-MAT-J03WOH 000002_DFL-MAT-J03WOY
000002_DFL-MAT-J03WPY 000002_DFL-MAT-J03WQQ 000002_DFL-MAT-J03WR9"
for match in $MATCHES; do
  for kind in DFL_02_01_matchinformation DFL_03_02_events_raw DFL_04_03_positions_raw_observed; do
    file="${kind}_DFL-COM-${match}.xml"
    curl -sSL -C - --retry 5 -o "$EXTERNAL/idsse/$file" "$IDSSE_BASE/$file?download=true"
  done
done
echo "Public data ready in $EXTERNAL"
