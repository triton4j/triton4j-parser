#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="build/samples/jar"
PACKAGE_NAME="org.triton4j.triton.test"
JAR_PATH="build/libs/dscope-triton-0.1.0.jar"

mkdir -p "$OUT_DIR"

./gradlew jar

java --add-modules jdk.incubator.code -jar "$JAR_PATH" \
  generate tutorials_python/02-fused-softmax.py \
  -o "$OUT_DIR" \
  -p "$PACKAGE_NAME" \
  -c FusedSoftmax

echo
echo "Generated files:"
find "$OUT_DIR" -type f | sort
