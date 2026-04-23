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

## TurboQuant Java Sample

This repository now also includes a Java-side TurboQuant sample under:

```text
src/main/java/org/triton4j/samples/turboquant/
```

What it includes:
- `TurboQuantCore`: pure Java quantize/dequantize implementation.
- `TurboQuantAttentionBenchmark`: Java attention microbenchmark with JSON report output.
- `TurboQuantTritonKernels`: Triton4j fused gather-dot and fused attention kernel entry points.
- `TurboQuantSample`: a runnable demo that compares fused-score math against dequantized reference scores.
- `TurboQuantBenchmarkCli`: a benchmark-only CLI for exporting JSON reports with configurable parameters.

This sample is useful because it connects several parts of the project in one place:
- pure Java reference math,
- Triton4j kernel entry points,
- benchmark report generation,
- HAT runtime execution tests against the same fused attention workload.

Current benchmark takeaway:
- TurboQuant gives strong compression and high cosine similarity,
- but the present Java fused implementation is still slower than the traditional Java baseline on CPU for the tested sizes,
- so these samples currently demonstrate correctness, compression, and scaling behavior more than raw CPU acceleration.

Run it with:

```bash
./gradlew runTurboQuantSample
```

The sample prints:
- fused vs dequantized quality metrics,
- theoretical compression ratio,
- benchmark timing summary,
- the generated report path.

Run the benchmark-only export task with defaults:

```bash
./gradlew benchmarkTurboQuant
```

Default report output:
- `build/reports/performance/turboquant-attention.json`

Run it with custom parameters:

```bash
./gradlew benchmarkTurboQuant -PturboQuantHeadDim=128 -PturboQuantSeqLen=512 -PturboQuantBits=3 -PturboQuantWarmupRuns=2 -PturboQuantMeasuredRuns=12 -PturboQuantSeed=7
```

Or run just the verification test:

```bash
./gradlew test --tests org.triton4j.samples.turboquant.TurboQuantSampleTest
```

Additional JUnit coverage:
- `org.triton4j.samples.turboquant.TurboQuantAttentionBenchmarkTest`
- `org.triton4j.samples.turboquant.TurboQuantTritonKernelTest`
- `org.triton4j.samples.turboquant.TurboQuantHatExecutionTest`
- `org.triton4j.samples.turboquant.TurboQuantGraalPyParityTest`
- `org.triton4j.samples.turboquant.TurboQuantComparisonTest`

The benchmark-oriented tests now cover:
- report generation for the benchmark JSON output,
- a size sweep over multiple `(headDim, seqLen, bits)` configurations,
- comparison of fused TurboQuant output against the traditional Java baseline.

## TurboQuant GraalPy Parity

There is also a GraalPy parity test for the TurboQuant sample that compares Java fused math against a pure-Python reference running inside GraalPy.

Run it with:

```bash
./gradlew test --tests org.triton4j.samples.turboquant.TurboQuantGraalPyParityTest
```

What it does:
- reuses deterministic TurboQuant sample inputs,
- compares fused scores and fused attention between Java and GraalPy,
- checks that `TurboQuantTritonKernels` still expose reflected Triton entry points.

Note:
- this is a GraalPy-compatible Python reference, not a direct execution of the original PyTorch/Triton Python files.

## TurboQuant HAT Runtime Validation

The TurboQuant sample also has a HAT runtime execution test that runs reflected fused attention on:
- `hat.backend.java.JavaSequentialBackend`
- `hat.backend.java.JavaMultiThreadedBackend`

Run it with:

```bash
./gradlew test --tests org.triton4j.samples.turboquant.TurboQuantHatExecutionTest
```

What it does:
- packs TurboQuant quantized keys and values into HAT buffers,
- dispatches the fused attention kernel through `hat.Accelerator.compute(...)`,
- compares HAT output against the Java fused TurboQuant reference,
- writes a report to `build/reports/performance/turboquant-hat-attention.json`.
