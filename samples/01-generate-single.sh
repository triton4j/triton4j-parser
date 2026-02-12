#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="build/samples/single"
PACKAGE_NAME="org.triton4j.triton.test"

mkdir -p "$OUT_DIR"

./gradlew run --args="generate tutorials_python/01-vector-add.py -o ${OUT_DIR} -p ${PACKAGE_NAME} -c VectorAdd"

echo
echo "Generated files:"
find "$OUT_DIR" -type f | sort
