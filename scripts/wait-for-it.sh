#!/usr/bin/env bash
# wait-for-it.sh: wait until a TCP host/port becomes available
# Source: https://github.com/vishnubob/wait-for-it (MIT License)
set -euo pipefail

TIMEOUT=30
QUIET=0
HOST=""
PORT=""
SLEEP_INTERVAL=1

usage() {
    cat <<USAGE >&2
Usage: $0 host:port [-s] [-t timeout] [-- command [args]]
  -h HOST | --host=HOST       Host or IP under test
  -p PORT | --port=PORT       TCP port under test
  -s | --strict               Only execute subcommand if the test succeeds
  -q | --quiet                Do not output any status messages
  -t TIMEOUT | --timeout=TIMEOUT
                              Timeout in seconds, zero for no timeout
USAGE
}

check_tcp() {
    if command -v nc >/dev/null 2>&1; then
        nc -z "$HOST" "$PORT" >/dev/null 2>&1
    else
        (echo > "/dev/tcp/${HOST}/${PORT}") >/dev/null 2>&1
    fi
}

wait_for() {
    if [[ $QUIET -eq 0 ]]; then
        echo "[wait-for-it] Waiting for ${HOST}:${PORT} with timeout ${TIMEOUT}s"
    fi
    local start_time end_time
    start_time=$(date +%s)
    while :; do
        if check_tcp; then
            if [[ $QUIET -eq 0 ]]; then
                end_time=$(date +%s)
                echo "[wait-for-it] Connection to ${HOST}:${PORT} succeeded after $((end_time - start_time))s"
            fi
            return 0
        fi
        if [[ $TIMEOUT -gt 0 ]]; then
            end_time=$(date +%s)
            if [[ $((end_time - start_time)) -ge $TIMEOUT ]]; then
                if [[ $QUIET -eq 0 ]]; then
                    echo "[wait-for-it] Timeout after ${TIMEOUT}s waiting for ${HOST}:${PORT}" >&2
                fi
                return 1
            fi
        fi
        sleep "$SLEEP_INTERVAL"
    done
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            *:*)
                HOST="${1%%:*}"
                PORT="${1##*:}"
                shift
                ;;
            -h)
                HOST="$2"
                shift 2
                ;;
            --host=*)
                HOST="${1#*=}"
                shift
                ;;
            -p)
                PORT="$2"
                shift 2
                ;;
            --port=*)
                PORT="${1#*=}"
                shift
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -q|--quiet)
                QUIET=1
                shift
                ;;
            -s|--strict)
                STRICT=1
                shift
                ;;
            --)
                shift
                COMMAND=("$@")
                break
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "[wait-for-it] Unknown argument: $1" >&2
                usage
                exit 1
                ;;
        esac
    done

    if [[ -z "$HOST" || -z "$PORT" ]]; then
        echo "[wait-for-it] Error: host and port must be specified" >&2
        usage
        exit 1
    fi
}

main() {
    STRICT=0
    COMMAND=()
    parse_args "$@"

    if wait_for; then
        if [[ ${#COMMAND[@]} -gt 0 ]]; then
            exec "${COMMAND[@]}"
        fi
    else
        if [[ $STRICT -eq 1 ]]; then
            exit 1
        fi
        if [[ ${#COMMAND[@]} -gt 0 ]]; then
            exec "${COMMAND[@]}"
        fi
    fi
}

main "$@"
