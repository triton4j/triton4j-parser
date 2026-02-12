#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="build/samples/all"
PACKAGE_NAME="org.triton4j.triton.test"

mkdir -p "$OUT_DIR"

./gradlew run --args="generate tutorials_python -o ${OUT_DIR} -p ${PACKAGE_NAME} --continue-on-error"

echo
echo "Generated files:"
find "$OUT_DIR" -type f | sort
