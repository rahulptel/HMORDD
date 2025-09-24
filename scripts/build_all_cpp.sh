#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(realpath "${SCRIPT_DIR}/..")"
SRC_CPP_DIR="${ROOT_DIR}/src/cpp"

usage() {
    cat <<'USAGE'
Usage: build_all_cpp.sh [options] [problem ...]

Options:
  -h, --help      Show this help message.

Specify one or more problem names (e.g. setpacking, tsp) to limit the build.
Without arguments, all available problems are built.
USAGE
}

FILTERS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -* )
            echo "[build_all_cpp] unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        * )
            FILTERS+=("$1")
            ;;
    esac
    shift
done

if [[ ! -d "${SRC_CPP_DIR}" ]]; then
    echo "[build_all_cpp] missing directory: ${SRC_CPP_DIR}" >&2
    exit 1
fi

mapfile -t make_scripts < <(find "${SRC_CPP_DIR}" -type f -name 'makelibcmd.sh' | sort)

if [[ ${#make_scripts[@]} -eq 0 ]]; then
    echo "[build_all_cpp] no makelibcmd.sh scripts found under ${SRC_CPP_DIR}" >&2
    exit 1
fi

should_build() {
    local problem="$1"
    if [[ ${#FILTERS[@]} -eq 0 ]]; then
        return 0
    fi
    for filter in "${FILTERS[@]}"; do
        if [[ "${problem}" == "${filter}" ]]; then
            return 0
        fi
    done
    return 1
}

selected=()
for script_path in "${make_scripts[@]}"; do
    problem="$(basename "$(dirname "${script_path}")")"
    if should_build "${problem}"; then
        selected+=("${script_path}")
    fi
done

if [[ ${#selected[@]} -eq 0 ]]; then
    if [[ ${#FILTERS[@]} -eq 0 ]]; then
        echo "[build_all_cpp] no makelibcmd.sh scripts matched" >&2
    else
        echo "[build_all_cpp] no makelibcmd.sh scripts matched filters: ${FILTERS[*]}" >&2
    fi
    exit 1
fi

for script_path in "${selected[@]}"; do
    problem="$(basename "$(dirname "${script_path}")")"
    rel_path="${script_path#${ROOT_DIR}/}"
    echo "[build_all_cpp] building ${problem} (${rel_path})"
    (
        cd "$(dirname "${script_path}")"
        bash "./$(basename "${script_path}")"
    )
    echo "[build_all_cpp] completed ${problem}"
done

echo "[build_all_cpp] all requested builds completed"
