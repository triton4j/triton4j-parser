# Samples

This directory contains runnable examples that show how to convert Triton Python tutorial files in `tutorials_python/` into Java using the CLI.

## Prerequisites

Run commands from the project root:

```bash
cd /path/to/TritonParser
```

Set a Babylon JDK:

```bash
export JAVA_HOME=$HOME/Java/babylon/build/macosx-aarch64-server-release/images/jdk
export PATH="$JAVA_HOME/bin:$PATH"
```

Optional macOS Metal profile:

```bash
export METAL_TC_BIN=/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v*/Metal.xctoolchain/usr/bin
export JAVA_HOME=$HOME/Java/babylon/build/macosx-aarch64-server-release/images/jdk
export PATH="$METAL_TC_BIN:$JAVA_HOME/bin:$PATH"
```

Note:
- `TritonParser` generation is backend-agnostic.
- Metal is used later by Babylon/Triton runtime backend stages.
- Keep Babylon JDK here because `jdk.incubator.code` is required.

## Sample Scripts

### 1) Single file conversion

Script:

```bash
./samples/01-generate-single.sh
```

What it does:
- Converts `tutorials_python/01-vector-add.py`
- Generates Java class `VectorAdd`
- Writes output to `build/samples/single/`

### 2) Convert all tutorial files

Script:

```bash
./samples/02-generate-all-tutorials.sh
```

What it does:
- Recursively scans `tutorials_python/`
- Converts all `.py` files
- Continues even if some files fail
- Writes output to `build/samples/all/`

### 3) Run from built jar

Script:

```bash
./samples/03-generate-from-jar.sh
```

What it does:
- Builds the project jar
- Runs the CLI via `java -jar`
- Converts one sample file
- Writes output to `build/samples/jar/`

## Direct CLI Commands (without scripts)

Single file:

```bash
./gradlew run --args="generate tutorials_python/01-vector-add.py -o build/samples/single -p org.triton4j.triton.test -c VectorAdd"
```

All tutorials:

```bash
./gradlew run --args="generate tutorials_python -o build/samples/all -p org.triton4j.triton.test --continue-on-error"
```

## Output Structure

Generated Java files are written into package directories under the output path.

Example:

```text
build/samples/single/org/triton4j/triton/test/VectorAdd.java
```

## Gradle Plugin Build Sample

There is also a dedicated Gradle subproject sample at:

```text
samples/plugin-build/
```

It applies `org.triton4j.codegen`, generates Java from a compilable subset of `tutorials_python/`, and compiles the generated sources.

Run:

```bash
./gradlew :samples-plugin-build:clean :samples-plugin-build:build
```
