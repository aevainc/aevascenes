#!/usr/bin/env bash
# Copyright (c) 2016-2025 Aeva, Inc.
# SPDX-License-Identifier: MIT
#
# The AevaScenes dataset is licensed separately under the AevaScenes Dataset License.
# See https://scenes.aeva.com/license for the full license text.

# download_dataset.sh - Download S3 presigned URLs safely with robust resume
# 
# This script downloads files from S3 presigned URLs while avoiding command line
# argument length limits. It supports multiple input methods, resumable downloads,
# automatic retries, and safe filename generation.
#
# SECURITY FEATURES:
# - Validates URLs before processing
# - Sanitizes filenames to prevent path traversal attacks
# - Uses temporary config files to avoid exposing URLs in process lists
# - Limits filename length to prevent filesystem issues
#
# DEPENDENCIES: curl or wget

set -euo pipefail

# Display usage information
usage() {
  cat <<'USAGE'
Usage:
  download_dataset.sh --url "https://...very-long-presigned-url..."
  download_dataset.sh --url-file path/to/url.txt
  echo "https://...long..." | download_dataset.sh --stdin

Options:
  -o, --output PATH         Save to this file (overrides automatic naming)
  -t, --timeout SEC         Network timeout in seconds (default: 300)
  -r, --retries N           Number of retry attempts for failed downloads (default: 5)
  --stdin                   Read URLs from stdin (one per line)
  --url URL                 Download a single URL provided as argument
  --url-file FILE           Read URLs from file (one per line)
  -q, --quiet               Suppress non-error output
  -h, --help                Show this help message

Examples:
  # Download single URL with custom output name
  bash scripts/download_dataset.sh --url "https://bucket.s3.amazonaws.com/file?X-Amz..." -o data/aevascenes_v0.1.tar.gz
  
  # Download multiple URLs from file
  download_dataset.sh --url-file urls.txt
  
  # Pipe URLs and download with custom timeout
  cat urls.txt | download_dataset.sh --stdin --timeout 600

Notes:
  - Presigned URLs contain authentication in query parameters - pass them as-is
  - URLs are processed via config files/stdin to avoid command line length limits
  - Downloads are automatically resumed if interrupted with robust retry logic
  - Filenames are sanitized for security and filesystem compatibility
USAGE
}

# Configuration defaults
OUTPUT_PATH=""
TIMEOUT=300
RETRIES=5
INPUT_MODE=""
URL_ARG=""
URL_FILE=""
QUIET=false

# Logging function that respects quiet mode
log() { 
    if ! "$QUIET"; then
        printf '%s\n' "$*" >&2
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output) 
            OUTPUT_PATH="${2-}"
            shift 2 
            ;;
        -t|--timeout) 
            TIMEOUT="${2-}"
            shift 2 
            ;;
        -r|--retries) 
            RETRIES="${2-}"
            shift 2 
            ;;
        --stdin) 
            INPUT_MODE="stdin"
            shift 
            ;;
        --url) 
            INPUT_MODE="arg"
            URL_ARG="${2-}"
            shift 2 
            ;;
        --url-file) 
            INPUT_MODE="file"
            URL_FILE="${2-}"
            shift 2 
            ;;
        -q|--quiet) 
            QUIET=true
            shift 
            ;;
        -h|--help) 
            usage
            exit 0 
            ;;
        *) 
            echo "Error: Unknown option '$1'" >&2
            usage
            exit 2 
            ;;
    esac
done

# Validate input mode was specified
if [[ -z "${INPUT_MODE}" ]]; then
    echo "Error: Must specify one of --stdin, --url, or --url-file" >&2
    usage
    exit 2
fi

# Detect available download tool
have_command() { 
    command -v "$1" >/dev/null 2>&1
}

    DOWNLOADER="wget"

log "Using $DOWNLOADER for downloads"

# util: trim whitespace and carriage returns
trim_whitespace() { 
    sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/\r$//' <<<"${1-}"
}

# util: convert %XX sequences to bytes and '+' to spaces
url_decode() {
    local encoded_string="${1//+/ }"
    printf '%b' "${encoded_string//%/\\x}"
}

# util: get filename from response-content-disposition parameter in query string
extract_filename_from_query() {
    local query_string="$1"
    local rcd_param disposition_value filename=""
    
    # Look for response-content-disposition parameter
    rcd_param="$(grep -oE '(^|&)response-content-disposition=[^&]*' <<<"$query_string" || true)"
    [[ -z "$rcd_param" ]] && return 1
    
    # Extract value after '=' and URL-decode it
    disposition_value="${rcd_param#*=}"
    disposition_value="$(url_decode "$disposition_value")"
    
    # Try filename*=charset'lang'encoded-value first
    filename="$(sed -nE "s/.*filename\\*=[^']*''([^;]+).*/\\1/p" <<<"$disposition_value")" || true
    
    if [[ -z "$filename" ]]; then
        # Fallback to standard format: filename="value" or filename=value
        filename="$(sed -nE 's/.*filename="?([^";]+)"?.*/\1/p' <<<"$disposition_value")" || true
    fi
    
    [[ -n "$filename" ]] && printf '%s' "$filename" || return 1
}

# util: generate safe filename from URL, query parameters over path
generate_safe_filename() {
    local url="$1"
    local path_part query_part filename=""
    
    # Split URL into path and query components
    if [[ "$url" == *\?* ]]; then
        query_part="${url#*\?}"
        path_part="${url%%\?*}"
    else
        query_part=""
        path_part="$url"
    fi
    
    # First try to extract filename from query parameters
    if [[ -n "$query_part" ]] && filename="$(extract_filename_from_query "$query_part")"; then
        log "Using filename from query parameters: $filename"
    else
        # Fallback: use last path segment
        local base_name="${path_part##*/}"
        filename="$(url_decode "$base_name")"
        [[ -n "$filename" ]] || filename="download"
        log "Using filename from URL path: $filename"
    fi
    
    # Sanitize filenames bad-chars
    local safe_filename
    safe_filename="$(printf '%s' "$filename" | LC_ALL=C tr -cd 'A-Za-z0-9.,_ -' | cut -c1-200)"
    
    # Check non-empty filename
    [[ -z "$safe_filename" ]] && safe_filename="download"
    
    printf '%s' "$safe_filename"
}

# Function to get file size
get_file_size() {
    local file="$1"
    if [[ -f "$file" ]]; then
        if have_command stat; then
            # Try GNU stat first, then BSD stat
            stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo "0"
        else
            # Fallback using ls
            ls -l "$file" | awk '{print $5}' 2>/dev/null || echo "0"
        fi
    else
        echo "0"
    fi
}


# Robust download function using wget with proper resume
download_with_wget() {
    local url="$1"
    local output_file="$2"
    local url_list_file
    local attempt=0
    local max_attempts=$((RETRIES + 1))
    local exit_code=1
    
    # Create temp file with URL
    url_list_file="$(mktemp)"
    printf '%s\n' "$url" >"$url_list_file"
    
    while [[ $attempt -lt $max_attempts ]]; do
        ((attempt++))
        
        local current_size
        current_size="$(get_file_size "$output_file")"
        
        if [[ $current_size -gt 0 ]]; then
            log "Resuming download from byte $current_size (attempt $attempt/$max_attempts)"
        else
            log "Starting fresh download (attempt $attempt/$max_attempts)"
        fi
        
        # Build wget args for this attempt
        local wget_args=( 
            "--timeout=30"
            "--read-timeout=${TIMEOUT}"
            "--tries=1"  # Handle retries ourselves
            "--continue"
            "--input-file=${url_list_file}"
            "--output-document=$output_file"
        )
        
        # Verbosity control
        if "$QUIET"; then
            wget_args+=("--quiet")
        else
            wget_args+=("--progress=bar:force")
        fi
        
        if wget "${wget_args[@]}"; then
            exit_code=0
            break
        else
            exit_code=$?
            log "Download attempt $attempt failed with exit code $exit_code"
            
            if [[ $attempt -lt $max_attempts ]]; then
                local wait_time=$((attempt * 2))
                log "Waiting ${wait_time}s before retry..."
                sleep "$wait_time"
            fi
        fi
    done
    
    # Cleanup temp list
    rm -f "$url_list_file"
    return "$exit_code"
}

# Main download dispatcher function
download_url() {
    local url="$1"
    local output_file=""
    
    # Determine output filename
    if [[ -n "$OUTPUT_PATH" ]]; then
        output_file="${OUTPUT_PATH}/$(generate_safe_filename "$url")"
    else
        output_file="$(generate_safe_filename "$url")"
    fi
    
    # Check if file already exists and is complete
    if [[ -f "$output_file" ]]; then
        local file_size
        file_size="$(get_file_size "$output_file")"
        if [[ $file_size -gt 0 ]]; then
            log "Found existing file: $output_file (${file_size} bytes)"
        fi
    fi
    
    download_with_wget "$url" "$output_file"
}

# Collect URLs based on input mode
declare -a URLS=()

case "$INPUT_MODE" in
    stdin)
        # Verify stdin has data available
        if [[ -t 0 ]]; then
            echo "Error: --stdin specified but no data available on stdin" >&2
            exit 2
        fi
        
        # Read URLs line by line from stdin
        while IFS= read -r line; do
            line="$(trim_whitespace "$line")"
            [[ -z "$line" ]] && continue  # Skip empty lines
            URLS+=("$line")
        done
        ;;
        
    arg)
        URLS+=("$(trim_whitespace "$URL_ARG")")
        ;;
        
    file)
        # Validate file exists and is readable
        if [[ -z "$URL_FILE" || ! -f "$URL_FILE" ]]; then
            echo "Error: URL file '$URL_FILE' not found or not readable" >&2
            exit 2
        fi
        
        # Read URLs line by line from file
        while IFS= read -r line; do
            line="$(trim_whitespace "$line")"
            [[ -z "$line" ]] && continue  # Skip empty lines
            URLS+=("$line")
        done < "$URL_FILE"
        ;;
esac

# Validate we have URLs to process
if [[ ${#URLS[@]} -eq 0 ]]; then
    echo "Error: No valid URLs provided" >&2
    exit 2
fi

log "Processing ${#URLS[@]} URL(s)"

# Main download loop with error handling
failed_downloads=0
for url in "${URLS[@]}"; do
    # Validate URL format
    if [[ ! "$url" =~ ^https?:// ]]; then
        log "Warning: Skipping invalid URL: $url"
        ((failed_downloads++))
        continue
    fi
    
    log "Downloading: ${url%%\?*}$([ "${url}" != "${url%%\?*}" ] && echo "?...")"
    
    if ! download_url "$url"; then
        log "Error: Failed to download $url after $((RETRIES + 1)) attempts"
        ((failed_downloads++))
    fi
done

if [[ $failed_downloads -eq 0 ]]; then
    log "All downloads completed successfully"
    
    # Provide extraction instructions for tar.gz files
    if ! "$QUIET"; then
        echo ""
        echo "To extract downloaded tar.gz files:"
        echo "  tar -xzf filename.tar.gz         # Extract to current directory"
        echo "  tar -xzf filename.tar.gz -C dir  # Extract to specific directory"
        echo "  tar -tzf filename.tar.gz         # List contents without extracting"
        echo ""
        echo "Alternative (two-step process):"
        echo "  gunzip filename.tar.gz           # Creates filename.tar"
        echo "  tar -xf filename.tar             # Extract the tar file"
    fi
    
    exit 0
else
    log "Warning: $failed_downloads download(s) failed"
    exit 1
fi