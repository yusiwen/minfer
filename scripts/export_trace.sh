#!/usr/bin/env bash
# Export and summarize a Metal System Trace (.trace) captured by Instruments.
#
# Background (measured 2026-08-13, M4 Pro / Xcode 26.6):
#   The correct xctrace lives INSIDE Xcode — /usr/bin/xctrace is a broken stub
#   that errors "tool not found". The Metal System Trace template records
#   per-forward GPU intervals (one row per command buffer = one minfer forward)
#   plus, when Counter Set is NOT null, a full GPU counter time-series that
#   reveals the per-phase bottleneck type (ALU-bound vs memory-bound).
#
#   Counter Set "null"                 -> NO counter sampling (shader list only)
#   Counter Set "Performance Limiters" -> 85 counters, full time series
#   Shader-Timeline intervals (per-kernel DURATIONS) stay EMPTY on this setup,
#   so the trace gives per-forward totals + bottleneck TYPE, not kernel times.
#
# Usage:
#   scripts/export_trace.sh <trace> [run]        export + summarize run (default: last)
#   scripts/export_trace.sh --list <trace>       list runs + recording config
#   scripts/export_trace.sh --raw <trace> <run> <schema>   dump one table to stdout
#
# Env:
#   XCT          xctrace binary (default: the one inside Xcode)
#   TRACE_OUT    output dir   (default: /tmp/minfer_trace_export)
#   TRACE_PROC   process name to filter per-forward intervals (default: minfer)
set -u

XCT="${XCT:-/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace}"
OUT="${TRACE_OUT:-/tmp/minfer_trace_export}"
PROC="${TRACE_PROC:-minfer}"
mkdir -p "$OUT"

toc() { "$XCT" export --input "$1" --toc 2>/dev/null; }

case "${1:-}" in
--list)
    [ -n "${2:-}" ] || { echo "usage: export_trace.sh --list <trace>" >&2; exit 1; }
    echo "=== runs ==="
    toc "$2" | sed -n 's/.*<run number="\([0-9]*\)">.*/run \1/p'
    echo "=== recording config (last run) ==="
    toc "$2" | grep -A6 'instrument name="Metal Application"' | grep -E 'Counter Set|Shader Timeline|Induced GPU' || true
    exit 0
    ;;
--raw)
    trace="${2:-}"; run="${3:-}"; schema="${4:-}"
    [ -n "$schema" ] || { echo "usage: export_trace.sh --raw <trace> <run> <schema>" >&2; exit 1; }
    "$XCT" export --input "$trace" \
        --xpath "/trace-toc/run[@number=\"$run\"]/data/table[@schema=\"$schema\"]"
    exit 0
    ;;
esac

trace="${1:-}"
[ -e "$trace" ] || { echo "no such trace: $trace" >&2; exit 1; }

run="${2:-}"
if [ -z "$run" ]; then
    run=$(toc "$trace" | sed -n 's/.*<run number="\([0-9]*\)">.*/\1/p' | tail -1)
fi
echo "=== run $run of $trace ==="
toc "$trace" | grep -A6 'instrument name="Metal Application"' | grep -E 'Counter Set|Shader Timeline|Induced GPU' || true

# Dump the tables. counter_value.xml can be hundreds of MB — keep it unless
# TRACE_KEEP_COUNTERS=1 (default: stream-parse, delete after).
dump() { "$XCT" export --input "$trace" \
    --xpath "/trace-toc/run[@number=\"$run\"]/data/table[@schema=\"$1\"]" 2>/dev/null > "$OUT/$2.xml"; }
dump gpu-counter-info               counter_info
dump metal-gpu-intervals            gpu_intervals
dump metal-shader-profiler-shader-list shader_list
dump metal-shader-profiler-intervals shader_intervals
if [ ! -f "$OUT/counter_value.xml" ]; then
    "$XCT" export --input "$trace" \
        --xpath "/trace-toc/run[@number=\"$run\"]/data/table[@schema=\"gpu-counter-value\"]" \
        2>/dev/null > "$OUT/counter_value.xml"
fi

# Python summary: counter map + per-phase bottleneck profile + per-forward times.
python3 - "$OUT" "$PROC" <<'PY'
import os, re, statistics, sys
out = sys.argv[1]
PROC = sys.argv[2]

def text(s):  # inner text of an element
    m = re.search(r'>(.*?)</', s)
    return m.group(1) if m else ''

# ---- counter map: id -> (name, description) ----
info = open(f"{out}/counter_info.xml").read()
rows = re.findall(r'<row>(.*?)</row>', info, re.S)
cmap = {}
for r in rows:
    mid = re.search(r'<uint32[^>]*>(\d+)</uint32>', r)
    mname = re.search(r'<gpu-counter-name[^>]*>(.*?)</gpu-counter-name>', r)
    mdesc = re.search(r'<string[^>]*>(.*?)</string>', r)
    if mid and mname:
        cmap[int(mid.group(1))] = {
            'name': mname.group(1),
            'desc': mdesc.group(1) if mdesc else '',
        }
print(f"counters: {len(cmap)}")

# ---- counter value time series (streamed, row-by-row) ----
# Row layout (single-line): <row><event-time..>ts</event-time><uint32..>cid</uint32>
#                            <fixed-decimal..>value</fixed-decimal>...
# Each timestamp samples ALL counters; group by sample-index to keep one
# snapshot per sampling tick.
row_re = re.compile(r'<row>(.*?)</row>')

agg = {}          # cid -> list of (ts_ns, value)
samples_ts = []   # list of distinct tick timestamps (ns)
ts_seen = set()
ts_lo = None
ts_hi = None
n_ticks = 0
val_path = f"{out}/counter_value.xml"
# xctrace dedups: the first row of each tick defines elements with
# <event-time id="N">ts</event-time> and <uint32 id="X" fmt="CID">CID</uint32>
# / <fixed-decimal id="Y" fmt="V">V</fixed-decimal>; later rows of the same tick
# and later ticks REFERENCE them (<event-time ref="N"/> / <uint32 ref="X"/> /
# <fixed-decimal ref="Y"/>). So counter-id = uint32 fmt (or ref->uid), value =
# fixed-decimal inline (or ref->uid). Track element-id -> value/counter maps.
ts_by_id = {}
uid_counter = {}
uid_value = {}
cur_ts = None
with open(val_path, errors='replace') as f:
    for line in f:
        m = row_re.search(line)
        if not m:
            continue
        body = m.group(1)
        mts_id = re.search(r'<event-time id="(\d+)"[^>]*>(\d+)</event-time>', body)
        mts_ref = re.search(r'<event-time ref="(\d+)"/>', body)
        if mts_id:
            ts_by_id[mts_id.group(1)] = int(mts_id.group(2))
            cur_ts = int(mts_id.group(2))
        elif mts_ref and mts_ref.group(1) in ts_by_id:
            cur_ts = ts_by_id[mts_ref.group(1)]
        if cur_ts is None:
            continue
        if ts_lo is None: ts_lo = cur_ts
        ts_hi = cur_ts
        n_ticks += 1
        if cur_ts not in ts_seen:
            ts_seen.add(cur_ts)
            samples_ts.append(cur_ts)
        # record element-id definitions
        for u in re.finditer(r'<uint32 id="(\d+)" fmt="(\d+)"[^>]*>', body):
            uid_counter[u.group(1)] = int(u.group(2))
        for v in re.finditer(r'<fixed-decimal id="(\d+)" fmt="([0-9eE.+-]+)"[^>]*>', body):
            uid_value[v.group(1)] = float(v.group(2))
        # resolve counter-id: inline fmt or ref
        mc_fmt = re.search(r'<uint32[^>]*fmt="(\d+)"[^>]*>(\d+)</uint32>', body)
        mc_ref = re.search(r'<uint32 ref="(\d+)"/>', body)
        cid = None
        if mc_fmt:
            cid = int(mc_fmt.group(1))
        elif mc_ref and mc_ref.group(1) in uid_counter:
            cid = uid_counter[mc_ref.group(1)]
        if cid is None:
            continue
        # resolve value: inline or ref
        val = None
        mv_fmt = re.search(r'<fixed-decimal[^>]*>([0-9eE.+-]+)</fixed-decimal>', body)
        mv_ref = re.search(r'<fixed-decimal ref="(\d+)"/>', body)
        if mv_fmt:
            val = float(mv_fmt.group(1))
        elif mv_ref and mv_ref.group(1) in uid_value:
            val = uid_value[mv_ref.group(1)]
        if val is None:
            continue
        agg.setdefault(cid, []).append((cur_ts, val))

print(f"counter rows: {n_ticks}  (distinct ticks: {len(samples_ts)})")
if ts_lo is not None:
    print(f"counter time window: {(ts_hi-ts_lo)/1e9:.2f} s")

print("\n=== bottleneck profile (whole run, percentage-type, % of peak) ===")
for cid in sorted(agg):
    if cid not in cmap:      # drop garbage ref-resolution keys (cid > 84)
        continue
    name = cmap[cid]['name']
    if not any(k in name for k in ('Limiter','Utilization','Occupancy','Residency','Target')):
        continue
    vals = [v for _, v in agg[cid]]
    m = statistics.mean(vals)
    if m < 0.5:
        continue
    print(f"  {name:<45} mean={m:7.1f}%  max={max(vals):7.1f}%  n={len(vals)}")

# ---- per-forward GPU intervals (PROC), ordered by start time ----
iv = open(f"{out}/gpu_intervals.xml").read()
irows = re.findall(r'<row>(.*?)</row>', iv, re.S)
forwards = []  # (start_ns, duration_us)
for r in irows:
    if PROC not in r:
        continue
    ms = re.search(r'<start-time id="\d+" fmt="[^"]*">(\d+)</start-time>', r)
    md = re.search(r'<duration id="\d+" fmt="[^"]*">(\d+)</duration>', r)
    if ms and md:
        forwards.append((int(ms.group(1)), int(md.group(1))/1e3))
forwards.sort()
if forwards:
    print(f"\n=== {PROC} forward GPU intervals: n={len(forwards)} ===")
    print(f"  prefill (first forward): {forwards[0][1]:.1f} us")
    dec = [d for _, d in forwards[1:]]
    if dec:
        print(f"  decode ({len(dec)} forwards): mean={statistics.mean(dec):.1f} us  median={statistics.median(dec):.1f} us  min={min(dec):.1f}  max={max(dec):.1f}")
    prefill_us = forwards[0][1]

# ---- per-kernel shader durations (metal-shader-profiler-intervals) ----
# Same id/ref dedup as the counter table: first occurrence of a duration /
# kernel-name / start-time defines an element id with a value; later rows
# reference it (<duration ref/> / <metal-object-label ref/> / <start-time ref/>).
# Single-target table (only the launched process appears), so no PROC filter.
si_path = f"{out}/shader_intervals.xml"
kernels = {}
if os.path.exists(si_path):
    si = open(si_path).read()
    dur_by_id = {}
    name_by_id = {}
    start_by_id = {}
    for r in re.findall(r'<row>(.*?)</row>', si, re.S):
        for m in re.finditer(r'<duration id="(\d+)"[^>]*>(\d+)</duration>', r):
            dur_by_id[m.group(1)] = int(m.group(2))
        for m in re.finditer(r'<metal-object-label id="(\d+)"[^>]*fmt="([^"]*)"', r):
            name_by_id[m.group(1)] = m.group(2)
        for m in re.finditer(r'<start-time id="(\d+)"[^>]*>(\d+)</start-time>', r):
            start_by_id[m.group(1)] = int(m.group(2))
        # resolve this row: name + duration + start
        mni = re.search(r'<metal-object-label[^>]*fmt="([^"]*)"', r)
        mnr = re.search(r'<metal-object-label ref="(\d+)"/>', r)
        mn = mni.group(1) if mni else (name_by_id.get(mnr.group(1)) if mnr else None)
        mdi = re.search(r'<duration[^>]*>(\d+)</duration>', r)
        mdr = re.search(r'<duration ref="(\d+)"/>', r)
        md = int(mdi.group(1)) if mdi else (dur_by_id.get(mdr.group(1)) if mdr else None)
        mp = re.search(r'<percent[^>]*>([0-9.]+)</percent>', r)
        if mn and md and 'kernel_' in mn:
            pct = float(mp.group(1)) if mp else 0.0
            kernels.setdefault(mn, {'n':0, 'us':0.0, 'pct':0.0})
            kernels[mn]['n'] += 1
            kernels[mn]['us'] += md/1e3
            kernels[mn]['pct'] += pct
if kernels:
    print(f"\n=== {PROC} per-kernel shader durations ({len(kernels)} kernels, {sum(k['n'] for k in kernels.values())} intervals) ===")
    print(f"  {'kernel':<48} {'n':>6} {'total_us':>10} {'avg_us':>8} {'%GPU':>6}")
    for name, k in sorted(kernels.items(), key=lambda kv: -kv[1]['us']):
        print(f"  {name:<48} {k['n']:>6} {k['us']:>10.1f} {k['us']/k['n']:>8.2f} {k['pct']:>6.1f}")

# ---- phase-split bottleneck profile ----
# Counter tick ts and forward start ts share the trace-relative ns timeline.
# prefill window = [first_forward_start, +prefill_duration]; decode = after.
if agg and forwards:
    t_pre = forwards[0][0] + int(forwards[0][1] * 1e3)  # prefill end (ns)
    phase_agg = {'prefill': {}, 'decode': {}}
    for cid, vals_ts in agg.items():
        if cid not in cmap:
            continue
        for ts, v in vals_ts:
            phase = 'prefill' if ts < t_pre else 'decode'
            phase_agg[phase].setdefault(cid, []).append(v)
    print("\n=== bottleneck profile by phase (percentage-type counters, % of peak) ===")
    for phase in ('prefill', 'decode'):
        print(f"\n-- {phase} --")
        rows = []
        for cid, vals in phase_agg[phase].items():
            name = cmap.get(cid, {}).get('name', '?')
            if not any(k in name for k in ('Limiter','Utilization','Occupancy','Residency','Target')):
                continue
            m = statistics.mean(vals)
            if m < 0.5:   # skip near-zero noise
                continue
            rows.append((name, m, max(vals)))
        rows.sort(key=lambda x: -x[1])
        for name, m, mx in rows[:12]:
            print(f"  {name:<45} mean={m:7.1f}%  max={mx:7.1f}%")
PY
