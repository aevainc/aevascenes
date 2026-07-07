#!/usr/bin/env bash
# Copyright (c) 2016-2025 Aeva, Inc.
# SPDX-License-Identifier: MIT
#
# Download AevaScenes presigned URLs with resume, retries, and parallel jobs.
# Dependency: wget

set -euo pipefail

# =============================================================================
# Defaults
# =============================================================================
OUTPUT_PATH=""
TIMEOUT=300
RETRIES=5
JOBS=8
INPUT_MODE=""
URL_ARG=""
URL_FILE=""
QUIET=false

TOTAL_URLS=0
START_EPOCH=0
declare -a URLS=()

# Temp files used while downloads run
FAIL_FILE=""
LAST_COMPLETED_FILE=""
COMPLETED_FILE=""
EXPECTED_BYTES_FILE=""
COMPLETE_LIST_FILE=""
PARTIAL_LIST_FILE=""
MANIFEST_FILE=""
ACTIVE_FILE=""
LOG_LOCK=""
MONITOR_ACTIVE=""
MONITOR_PID=""

# =============================================================================
# Usage
# =============================================================================
usage() {
    cat <<'EOF'
Usage:
  download_dataset.sh --url-file urls.txt --output data/aevascenes_v2
  download_dataset.sh --url "https://..."
  cat urls.txt | download_dataset.sh --stdin --output data/aevascenes_v2

Options:
  -o, --output PATH   Output directory (files go to train/validation/test subdirs when present in URL)
  -j, --jobs N        Parallel downloads (default: 8)
  -r, --retries N     Retries per file (default: 5)
  -t, --timeout SEC   Read timeout in seconds (default: 300)
  --url-file FILE     One URL per line
  --url URL           Single URL
  --stdin             Read URLs from stdin
  -q, --quiet         Errors only
  -h, --help          Show this help

Notes:
  Re-running the script is safe. Partial downloads resume automatically.
EOF
}

# =============================================================================
# Logging and formatting
# =============================================================================
log() {
    [[ "$QUIET" == true ]] && return
    if [[ -n "$LOG_LOCK" ]]; then
        flock "$LOG_LOCK" printf '\n%s\n' "$*" >&2
    else
        printf '\n%s\n' "$*" >&2
    fi
}

format_bytes() {
    awk -v b="${1:-0}" 'BEGIN {
        if (b >= 1073741824) printf "%.2f GB", b/1073741824
        else if (b >= 1048576) printf "%.1f MB", b/1048576
        else if (b >= 1024) printf "%.1f KB", b/1024
        else printf "%d B", b
    }'
}

format_duration() {
    local s="${1:-0}" h m
    h=$((s / 3600)); m=$(((s % 3600) / 60)); s=$((s % 60))
    if ((h > 0)); then printf '%dh %02dm' "$h" "$m"
    elif ((m > 0)); then printf '%dm %02ds' "$m" "$s"
    else printf '%ds' "$s"; fi
}

format_speed() {
    local bytes="${1:-0}" seconds="${2:-1}"
    ((seconds < 1)) && seconds=1
    format_bytes $((bytes / seconds))
    printf '/s'
}

# Estimate time remaining from bytes left and download speed
format_eta() {
    local remaining="${1:-0}" speed="${2:-0}" avg_speed="${3:-0}" eta=0
    ((remaining <= 0)) && { printf 'ETA done'; return; }
    if ((speed > 0)); then
        eta=$((remaining / speed))
    elif ((avg_speed > 0)); then
        eta=$((remaining / avg_speed))
    else
        printf 'ETA --'
        return
    fi
    printf 'ETA %s' "$(format_duration "$eta")"
}

short_name() {
    local name="${1##*/}"
    ((${#name} > 44)) && printf '%s...%s' "${name:0:18}" "${name: -21}" || printf '%s' "$name"
}

progress_bar() {
    local current="${1:-0}" total="${2:-1}" width=20 filled=0 i out=""
    ((total > 0)) && filled=$((current * width / total))
    ((filled > width)) && filled=$width
    out='['
    for ((i = 0; i < width; i++)); do
        if ((i < filled)); then out+='='
        elif ((i == filled && filled < width)); then out+='>'
        else out+=' '; fi
    done
    printf '%s]' "$out"
}

# =============================================================================
# Shared counters (safe across parallel jobs)
# =============================================================================
read_counter() { cat "$1"; }

add_counter() {
    local file="$1" delta="${2:-1}"
    flock "$LOG_LOCK" bash -c '
        n=$(( $(cat "$1") + $2 ))
        (( n < 0 )) && n=0
        echo "$n" > "$1"
    ' _ "$file" "$delta"
}

# Log a finished file and update the running completion count
log_completion() {
    local index="$1" status="$2" name="$3" extra="${4:-}"
    local done left

    add_counter "$COMPLETED_FILE" 1
    flock "$LOG_LOCK" printf '%s\n' "$name" >"$LAST_COMPLETED_FILE"
    done=$(read_counter "$COMPLETED_FILE")
    left=$((TOTAL_URLS - done))

    if [[ -n "$extra" ]]; then
        log "✓  ${done}/${TOTAL_URLS}  #$(printf '%03d' "$index")  ${status}  ${name}  ${extra}  ·  ${left} left"
    else
        log "✓  ${done}/${TOTAL_URLS}  #$(printf '%03d' "$index")  ${status}  ${name}  ·  ${left} left"
    fi
}

parallel_mode_label() {
    if ((JOBS == 1)); then
        printf 'sequential (1 at a time)'
    else
        printf 'up to %d files in parallel' "$JOBS"
    fi
}

# =============================================================================
# URL helpers: filename, split, output path
# =============================================================================
trim() {
    sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/\r$//' <<<"${1-}"
}

url_path() { printf '%s' "${1%%\?*}"; }

filename_from_url() {
    local url="$1" path query name
    path="$(url_path "$url")"
    name="${path##*/}"
    name="$(printf '%b' "${name//+/ }")"
    name="$(printf '%s' "$name" | LC_ALL=C tr -cd 'A-Za-z0-9.,_ -' | cut -c1-200)"
    [[ -n "$name" ]] || name="download"
    printf '%s' "$name"
}

split_from_url() {
    local path="$1"
    [[ "$path" =~ /(train|validation|test)/ ]] && printf '%s' "${BASH_REMATCH[1]}"
}

output_path_for() {
    local url="$1" split name
    name="$(filename_from_url "$url")"
    if [[ -z "$OUTPUT_PATH" ]]; then
        printf '%s' "$name"
        return
    fi
    split="$(split_from_url "$(url_path "$url")")"
    if [[ -n "$split" ]]; then
        printf '%s/%s/%s' "$OUTPUT_PATH" "$split" "$name"
    else
        printf '%s/%s' "$OUTPUT_PATH" "$name"
    fi
}

# =============================================================================
# Load URLs from stdin, file, or single argument
# =============================================================================
read_urls() {
    local line
    case "$INPUT_MODE" in
        stdin)
            [[ -t 0 ]] && { echo "Error: --stdin but no input on stdin" >&2; exit 2; }
            while IFS= read -r line || [[ -n "${line-}" ]]; do
                line="$(trim "$line")"
                [[ -n "$line" ]] && URLS+=("$line")
            done
            ;;
        arg)
            URLS+=("$(trim "$URL_ARG")")
            ;;
        file)
            [[ -f "$URL_FILE" ]] || { echo "Error: file not found: $URL_FILE" >&2; exit 2; }
            while IFS= read -r line || [[ -n "${line-}" ]]; do
                line="$(trim "$line")"
                [[ -n "$line" ]] && URLS+=("$line")
            done <"$URL_FILE"
            ;;
    esac
}

# =============================================================================
# Size probes and on-disk totals
# =============================================================================
remote_size() {
    wget --timeout=15 --spider --server-response "$1" 2>&1 \
        | awk '/^[[:space:]]*[Cc]ontent-[Ll]ength:/ { len=$2 } END { print len+0 }'
}

probe_expected_sizes() {
    local url
    for url in "${URLS[@]}"; do
        [[ "$url" =~ ^https?:// ]] || continue
        wait_for_job_slot
        (
            local size dest
            size="$(remote_size "$url")"
            dest="$(output_path_for "$url")"
            if ((size > 0)); then
                flock "$LOG_LOCK" printf '%s\t%s\n' "$size" "$dest" >>"$MANIFEST_FILE"
                add_counter "$EXPECTED_BYTES_FILE" "$size"
            fi
        ) &
    done
    wait || true
}

expected_size_for() {
    awk -F'\t' -v d="$1" '$2 == d { print $1; exit }' "$MANIFEST_FILE"
}

# Scan disk, classify sequences, and print resume summary
show_resume_status() {
    local url dest local_size remote_size name
    local complete=0 partial=0 fresh=0 already_bytes=0

    : >"$COMPLETE_LIST_FILE"
    : >"$PARTIAL_LIST_FILE"

    for url in "${URLS[@]}"; do
        [[ "$url" =~ ^https?:// ]] || continue
        dest="$(output_path_for "$url")"
        name="$(short_name "$dest")"
        local_size="$(file_size "$dest")"
        remote_size="$(expected_size_for "$dest")"

        if ((local_size > 0 && remote_size > 0 && local_size >= remote_size)); then
            complete=$((complete + 1))
            already_bytes=$((already_bytes + local_size))
            printf '%s\n' "$name" >>"$COMPLETE_LIST_FILE"
        elif ((local_size > 0)); then
            partial=$((partial + 1))
            already_bytes=$((already_bytes + local_size))
            printf '%s\t%s\t%s\n' "$name" "$local_size" "$remote_size" >>"$PARTIAL_LIST_FILE"
        else
            fresh=$((fresh + 1))
        fi
    done

    echo "$complete" >"$COMPLETED_FILE"

    log "  resume: enabled — safe to re-run; partial files continue where they left off"

    if ((complete > 0 || partial > 0)); then
        log "  on disk: $complete complete, $partial partial, $fresh new ($(format_bytes "$already_bytes") already downloaded)"
    else
        log "  on disk: no existing files — starting fresh"
        return
    fi

    if ((complete > 0)); then
        log "  complete sequences:"
        while IFS= read -r name; do
            [[ -n "$name" ]] && log "    ✓ $name"
        done < <(head -10 "$COMPLETE_LIST_FILE") || true
        if ((complete > 10)); then
            log "    ... and $((complete - 10)) more"
        fi
    fi

    if ((partial > 0)); then
        log "  partial sequences (will resume):"
        while IFS=$'\t' read -r name local_size remote_size; do
            [[ -n "$name" ]] && log "    ↻ $name  ($(format_bytes "$local_size") / $(format_bytes "$remote_size"))"
        done < <(head -10 "$PARTIAL_LIST_FILE") || true
        if ((partial > 10)); then
            log "    ... and $((partial - 10)) more"
        fi
    fi
}

is_already_complete() {
    local url="$1" dest local_size remote_size
    dest="$(output_path_for "$url")"
    local_size="$(file_size "$dest")"
    remote_size="$(expected_size_for "$dest")"
    ((local_size > 0 && remote_size > 0 && local_size >= remote_size))
}

bytes_on_disk() {
    local root="${OUTPUT_PATH:-.}" total=0 size
    [[ -d "$root" ]] || { echo 0; return; }
    while IFS= read -r size; do
        total=$((total + size))
    done < <(find "$root" -type f -name '*.tar.gz' -printf '%s\n' 2>/dev/null)
    echo "$total"
}

file_size() {
    [[ -f "$1" ]] && stat -c%s "$1" 2>/dev/null || echo 0
}

# =============================================================================
# Live progress line (updates once per second)
# =============================================================================
progress_monitor() {
    local last_bytes=0 last_time="$START_EPOCH"
    while [[ -f "$MONITOR_ACTIVE" ]]; do
        sleep 1
        [[ -f "$MONITOR_ACTIVE" ]] || break

        local now elapsed bytes expected done active pct bar speed dt delta size_str parallel_str
        local remaining avg_speed eta_str files_left last_name
        now=$(date +%s)
        elapsed=$((now - START_EPOCH))
        bytes=$(bytes_on_disk)
        expected=$(read_counter "$EXPECTED_BYTES_FILE")
        done=$(read_counter "$COMPLETED_FILE")
        active=$(read_counter "$ACTIVE_FILE")
        files_left=$((TOTAL_URLS - done))
        last_name="$(cat "$LAST_COMPLETED_FILE" 2>/dev/null || echo '—')"
        last_name="$(short_name "$last_name")"

        if ((JOBS == 1)); then
            parallel_str="sequential"
        else
            parallel_str="${active}/${JOBS} parallel"
        fi

        if ((expected > 0)); then
            pct=$((bytes * 100 / expected)); ((pct > 100)) && pct=100
            bar="$(progress_bar "$bytes" "$expected")"
            size_str="$(format_bytes "$bytes") / $(format_bytes "$expected")"
            remaining=$((expected - bytes))
            ((remaining < 0)) && remaining=0
        else
            pct=0; ((TOTAL_URLS > 0)) && pct=$((done * 100 / TOTAL_URLS))
            bar="$(progress_bar "$done" "$TOTAL_URLS")"
            size_str="$(format_bytes "$bytes") / ?"
            remaining=0
        fi

        dt=$((now - last_time))
        delta=$((bytes - last_bytes))
        speed=0
        ((dt > 0 && delta >= 0)) && speed=$((delta / dt))
        avg_speed=0
        ((elapsed > 0)) && avg_speed=$((bytes / elapsed))
        last_bytes=$bytes
        last_time=$now
        eta_str="$(format_eta "$remaining" "$speed" "$avg_speed")"

        flock "$LOG_LOCK" printf '\r\033[K  %s %3d%%  %s  ·  %s/s  ·  %d done · %d left · %s  ·  %s  ·  last: %s' \
            "$bar" "$pct" "$size_str" "$(format_bytes "$speed")" "$done" "$files_left" \
            "$parallel_str" "$eta_str" "$last_name" >&2
    done
    flock "$LOG_LOCK" printf '\n' >&2
}

stop_progress_monitor() {
    rm -f "$MONITOR_ACTIVE"
    [[ -n "$MONITOR_PID" ]] && wait "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
}

# =============================================================================
# Download one file with wget (resume + retries)
# =============================================================================
download_file() {
    local url="$1" dest="$2"
    local attempt=0 max=$((RETRIES + 1)) wait_s list
    list="$(mktemp)"
    printf '%s\n' "$url" >"$list"

    while ((attempt < max)); do
        attempt=$((attempt + 1))
        if wget --timeout=30 --read-timeout="$TIMEOUT" --tries=1 --continue --quiet \
            --input-file="$list" --output-document="$dest"; then
            rm -f "$list"
            return 0
        fi
        if ((attempt < max)); then
            wait_s=$((attempt * 2))
            log "  retry $attempt/$max in ${wait_s}s for $(short_name "$dest")"
            sleep "$wait_s"
        fi
    done

    rm -f "$list"
    return 1
}

download_one() {
    local url="$1" index="$2"
    local dest name t0 t1 elapsed size partial expected tag=""

    add_counter "$ACTIVE_FILE" 1
    trap 'add_counter "$ACTIVE_FILE" -1' EXIT
    ((JOBS > 1)) && tag=" (parallel)"

    dest="$(output_path_for "$url")"
    [[ "$dest" == */* ]] && mkdir -p "$(dirname "$dest")"
    name="$(short_name "$dest")"
    # Already complete — counted at startup, no download needed
    if is_already_complete "$url"; then
        return 0
    fi

    partial="$(file_size "$dest")"
    expected="$(expected_size_for "$dest")"

    if ((partial > 0)); then
        log "[$(printf '%4d' "$index")/$TOTAL_URLS] resume${tag} $name ($(format_bytes "$partial") / $(format_bytes "$expected"))"
    elif [[ "$QUIET" == false && JOBS -eq 1 ]]; then
        log "[$(printf '%4d' "$index")/$TOTAL_URLS] start${tag}  $name"
    fi

    t0=$(date +%s)
    if download_file "$url" "$dest"; then
        t1=$(date +%s)
        elapsed=$((t1 - t0))
        size="$(file_size "$dest")"
        log_completion "$index" "done" "$name" "$(format_bytes "$size") $(format_duration "$elapsed") $(format_speed "$size" "$elapsed")"
        return 0
    fi

    log "[$(printf '%4d' "$index")/$TOTAL_URLS] FAIL   $name"
    echo 1 >>"$FAIL_FILE"
    return 1
}

# =============================================================================
# Parallel job pool
# =============================================================================
wait_for_job_slot() {
    while (( $(jobs -rp | wc -l) >= JOBS )); do
        wait -n 2>/dev/null || wait || true
    done
}

run_downloads() {
    local url index=0 failed=0 pending=0

    for url in "${URLS[@]}"; do
        if [[ ! "$url" =~ ^https?:// ]]; then
            log "[warn] skipping invalid URL: ${url:0:60}..."
            failed=$((failed + 1))
            continue
        fi
        ((index++)) || true
        if is_already_complete "$url"; then
            continue
        fi
        pending=$((pending + 1))
        wait_for_job_slot
        download_one "$url" "$index" &
    done

    if ((pending == 0)); then
        log "  all sequences already complete — nothing to download"
    else
        log "  fetching $pending remaining sequences ($(parallel_mode_label))..."
    fi

    wait || true
    [[ -s "$FAIL_FILE" ]] && failed=$((failed + $(wc -l <"$FAIL_FILE")))
    echo "$failed"
}

# =============================================================================
# Parse CLI arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output) OUTPUT_PATH="${2-}"; shift 2 ;;
        -t|--timeout) TIMEOUT="${2-}"; shift 2 ;;
        -r|--retries) RETRIES="${2-}"; shift 2 ;;
        -j|--jobs) JOBS="${2-}"; shift 2 ;;
        --stdin) INPUT_MODE="stdin"; shift ;;
        --url) INPUT_MODE="arg"; URL_ARG="${2-}"; shift 2 ;;
        --url-file) INPUT_MODE="file"; URL_FILE="${2-}"; shift 2 ;;
        -q|--quiet) QUIET=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

[[ -n "$INPUT_MODE" ]] || { echo "Error: use --stdin, --url, or --url-file" >&2; usage; exit 2; }
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "Error: --jobs must be a positive integer" >&2; exit 2; }

# =============================================================================
# Main
# =============================================================================
read_urls
[[ ${#URLS[@]} -gt 0 ]] || { echo "Error: no URLs to download" >&2; exit 2; }
[[ -n "$OUTPUT_PATH" ]] && mkdir -p "$OUTPUT_PATH"

TOTAL_URLS=${#URLS[@]}
START_EPOCH=$(date +%s)

FAIL_FILE="$(mktemp)"
COMPLETED_FILE="$(mktemp)"
EXPECTED_BYTES_FILE="$(mktemp)"
LOG_LOCK="$(mktemp)"
ACTIVE_FILE="$(mktemp)"
MANIFEST_FILE="$(mktemp)"
COMPLETE_LIST_FILE="$(mktemp)"
PARTIAL_LIST_FILE="$(mktemp)"
LAST_COMPLETED_FILE="$(mktemp)"
MONITOR_ACTIVE="$(mktemp)"
echo 0 >"$COMPLETED_FILE"
echo 0 >"$EXPECTED_BYTES_FILE"
echo 0 >"$ACTIVE_FILE"
printf '—\n' >"$LAST_COMPLETED_FILE"
trap 'stop_progress_monitor; rm -f "$FAIL_FILE" "$COMPLETED_FILE" "$EXPECTED_BYTES_FILE" "$MANIFEST_FILE" "$COMPLETE_LIST_FILE" "$PARTIAL_LIST_FILE" "$ACTIVE_FILE" "$LAST_COMPLETED_FILE" "$LOG_LOCK" "$MONITOR_ACTIVE"' EXIT

# Banner
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "  AevaScenes download"
log "  files: $TOTAL_URLS  |  mode: $(parallel_mode_label)  |  retries: $RETRIES"
log "  output: ${OUTPUT_PATH:-$(pwd)}"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Measure total remote size, then start live progress
if [[ "$QUIET" == false ]]; then
    log "  measuring remote sizes..."
fi
probe_expected_sizes
if [[ "$QUIET" == false ]]; then
    log "  expected total: $(format_bytes "$(read_counter "$EXPECTED_BYTES_FILE")")"
    show_resume_status
    progress_monitor &
    MONITOR_PID=$!
fi

failed_downloads=$(run_downloads)
stop_progress_monitor

# Summary
elapsed=$(($(date +%s) - START_EPOCH))
downloaded=$(bytes_on_disk)
expected=$(read_counter "$EXPECTED_BYTES_FILE")
succeeded=$((TOTAL_URLS - failed_downloads))

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if ((failed_downloads == 0)); then
    log "  complete: $succeeded/$TOTAL_URLS files  |  $(format_bytes "$downloaded") / $(format_bytes "$expected")  |  $(format_duration "$elapsed")  |  avg $(format_speed "$downloaded" "$elapsed")"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [[ "$QUIET" == false ]]; then
        echo ""
        echo "Extract archives:"
        echo "  cd ${OUTPUT_PATH:-.} && for split in train validation test; do"
        echo "    [ -d \"\$split\" ] && for f in \"\$split\"/*.tar.gz; do tar -xzf \"\$f\" -C \"\$split\"; done"
        echo "  done"
    fi
    exit 0
fi

log "  finished with errors: $succeeded/$TOTAL_URLS succeeded, $failed_downloads failed  |  $(format_duration "$elapsed")"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
exit 1
