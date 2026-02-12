# Plugin Build Sample

This sample demonstrates a Gradle build that:
- applies the local `org.triton4j.codegen` plugin,
- generates Java from a compilable subset of `tutorials_python/*.py`,
- compiles the generated Java sources.

Included tutorials:
- `01-vector-add.py`
- `02-fused-softmax.py`
- `03-matrix-multiplication.py`
- `04-low-memory-dropout.py`
- `05-layer-norm.py`
- `07-extern-functions.py`
- `08-grouped-gemm.py`

## Run

From project root:

```bash
./gradlew :samples-plugin-build:clean :samples-plugin-build:build
```

## Important outputs

Generated Java:

```text
samples/plugin-build/build/generated-triton-java/
```

Compiled generated classes:

```text
samples/plugin-build/build/classes/generated-triton-java/
```

## Useful tasks

Generate only:

```bash
./gradlew :samples-plugin-build:generateTritonJava
```

Compile generated Java only:

```bash
./gradlew :samples-plugin-build:compileGeneratedTritonJava
```
