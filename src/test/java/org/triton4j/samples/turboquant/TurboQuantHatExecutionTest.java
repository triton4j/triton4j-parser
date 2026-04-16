package org.triton4j.samples.turboquant;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.Locale;
import java.util.Random;

import org.junit.jupiter.api.Test;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.F32Array;
import hat.buffer.S32Array;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface.RO;
import optkl.ifacemapper.MappableIface.WO;

public class TurboQuantHatExecutionTest {

	private static final String JAVA_SEQ_BACKEND = "hat.backend.java.JavaSequentialBackend";
	private static final String JAVA_MT_BACKEND = "hat.backend.java.JavaMultiThreadedBackend";

	private static final int HEAD_DIM = 64;
	private static final int SEQ_LEN = 96;
	private static final int BITS = 4;
	private static final long SEED = 23L;
	private static final int WARMUP_RUNS = 2;
	private static final int MEASURED_RUNS = 5;
	private static final Path REPORT_DIR = Paths.get("build", "reports", "performance");
	private static final Path REPORT_FILE = REPORT_DIR.resolve("turboquant-hat-attention.json");

	@Reflect
	public static void fusedAttentionOutputKernel(
			@RO KernelContext kc,
			@RO F32Array rotatedQuery,
			@RO S32Array keyIndices,
			@RO F32Array keyNorms,
			@RO F32Array centroids,
			@RO F32Array values,
			@WO F32Array out,
			int seqLen,
			int headDim,
			float scale) {
		int outputIndex = kc.gix;
		if (outputIndex >= headDim)
			return;

		float maxScore = Float.NEGATIVE_INFINITY;
		for (int seq = 0; seq < seqLen; seq++) {
			long base = (long) seq * headDim;
			float dot = 0.0f;
			for (int dim = 0; dim < headDim; dim++) {
				int centroidIndex = keyIndices.array(base + dim);
				dot += rotatedQuery.array(dim) * centroids.array(centroidIndex);
			}
			float score = keyNorms.array(seq) * dot * scale;
			if (score > maxScore)
				maxScore = score;
		}

		float weightedSum = 0.0f;
		float normalizer = 0.0f;
		for (int seq = 0; seq < seqLen; seq++) {
			long base = (long) seq * headDim;
			float dot = 0.0f;
			for (int dim = 0; dim < headDim; dim++) {
				int centroidIndex = keyIndices.array(base + dim);
				dot += rotatedQuery.array(dim) * centroids.array(centroidIndex);
			}
			float score = keyNorms.array(seq) * dot * scale;
			float weight = (float) Math.exp(score - maxScore);
			normalizer += weight;
			weightedSum += weight * values.array(base + outputIndex);
		}

		out.array(outputIndex, weightedSum / Math.max(normalizer, 1.0e-8f));
	}

	@Reflect
	public static void fusedAttention(
			@RO ComputeContext cc,
			@RO F32Array rotatedQuery,
			@RO S32Array keyIndices,
			@RO F32Array keyNorms,
			@RO F32Array centroids,
			@RO F32Array values,
			@WO F32Array out,
			int seqLen,
			int headDim,
			float scale) {
		cc.dispatchKernel(
				NDRange.of1D(headDim),
				kc -> fusedAttentionOutputKernel(kc, rotatedQuery, keyIndices, keyNorms, centroids, values, out, seqLen,
						headDim, scale));
	}

	@Test
	void executesFusedAttentionOnHatBackends() throws Exception {
		BackendResult seq = benchmarkBackend(JAVA_SEQ_BACKEND);
		BackendResult mt = benchmarkBackend(JAVA_MT_BACKEND);

		assertTrue(seq.avgNanos() > 0L, "Sequential backend produced no timing.");
		assertTrue(mt.avgNanos() > 0L, "Multithreaded backend produced no timing.");
		assertTrue(seq.comparison().maxAbsDiff() < 1.0e-4f,
				"Sequential HAT output diverged from TurboQuant fused attention reference: maxDiff="
						+ seq.comparison().maxAbsDiff());
		assertTrue(mt.comparison().maxAbsDiff() < 1.0e-4f,
				"Multithreaded HAT output diverged from TurboQuant fused attention reference: maxDiff="
						+ mt.comparison().maxAbsDiff());
		assertTrue(seq.comparison().cosineSimilarity() > 0.9999f,
				"Sequential HAT output cosine similarity too low: " + seq.comparison().cosineSimilarity());
		assertTrue(mt.comparison().cosineSimilarity() > 0.9999f,
				"Multithreaded HAT output cosine similarity too low: " + mt.comparison().cosineSimilarity());

		double speedup = (double) seq.avgNanos() / (double) mt.avgNanos();
		System.out.printf(Locale.ROOT,
				"HAT TurboQuant fused attention (headDim=%d, seqLen=%d): seq=%.3f ms, mt=%.3f ms, speedup(seq/mt)=%.3fx%n",
				HEAD_DIM,
				SEQ_LEN,
				nanosToMillis(seq.avgNanos()),
				nanosToMillis(mt.avgNanos()),
				speedup);
		Path report = writeReport(seq, mt, speedup);
		assertTrue(Files.exists(report), "Expected HAT TurboQuant report to exist: " + report.toAbsolutePath());
	}

	@Reflect
	public static BackendResult benchmarkBackend(String backendClassName) {
		TurboQuantCore turboQuant = new TurboQuantCore(HEAD_DIM, BITS, SEED);
		Random random = new Random(SEED);
		float[] query = TurboQuantSample.randomVector(random, HEAD_DIM);
		float[][] keys = TurboQuantSample.randomMatrix(random, SEQ_LEN, HEAD_DIM);
		float[][] valuesMatrix = TurboQuantSample.randomMatrix(random, SEQ_LEN, HEAD_DIM);
		TurboQuantCore.QuantizedVectors quantizedKeys = turboQuant.quantize(keys);
		float[] rotatedQuery = turboQuant.rotate(query);
		float[] centroids = turboQuant.centroids();
		float[] keyNorms = quantizedKeys.norms();
		int[] flatIndices = flatten(quantizedKeys.indices());
		float[] flatValues = flatten(valuesMatrix);
		float scale = TurboQuantCore.attentionScale(HEAD_DIM);
		float[] expected = turboQuant.fusedAttention(rotatedQuery, quantizedKeys, valuesMatrix, scale);

		Accelerator accelerator = new Accelerator(MethodHandles.lookup(),
				backend -> backend.getClass().getName().equals(backendClassName));

		F32Array rotatedQueryBuffer = F32Array.createFrom(accelerator, rotatedQuery);
		S32Array keyIndicesBuffer = S32Array.createFrom(accelerator, flatIndices);
		F32Array keyNormsBuffer = F32Array.createFrom(accelerator, keyNorms);
		F32Array centroidsBuffer = F32Array.createFrom(accelerator, centroids);
		F32Array valuesBuffer = F32Array.createFrom(accelerator, flatValues);
		F32Array outBuffer = F32Array.create(accelerator, HEAD_DIM);

		for (int run = 0; run < WARMUP_RUNS; run++)
			accelerator.compute(cc -> fusedAttention(cc, rotatedQueryBuffer, keyIndicesBuffer, keyNormsBuffer,
					centroidsBuffer, valuesBuffer, outBuffer, SEQ_LEN, HEAD_DIM, scale));

		long totalNanos = 0L;
		for (int run = 0; run < MEASURED_RUNS; run++) {
			long start = System.nanoTime();
			accelerator.compute(cc -> fusedAttention(cc, rotatedQueryBuffer, keyIndicesBuffer, keyNormsBuffer,
					centroidsBuffer, valuesBuffer, outBuffer, SEQ_LEN, HEAD_DIM, scale));
			totalNanos += System.nanoTime() - start;
		}

		float[] actual = outBuffer.arrayView();
		TurboQuantCore.Comparison comparison = TurboQuantCore.compare(expected, actual);
		return new BackendResult(backendClassName, totalNanos / MEASURED_RUNS, comparison);
	}

	private static int[] flatten(int[][] matrix) {
		int rows = matrix.length;
		int columns = rows == 0 ? 0 : matrix[0].length;
		int[] flat = new int[rows * columns];
		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++)
				flat[row * columns + column] = matrix[row][column];
		}
		return flat;
	}

	private static float[] flatten(float[][] matrix) {
		int rows = matrix.length;
		int columns = rows == 0 ? 0 : matrix[0].length;
		float[] flat = new float[rows * columns];
		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++)
				flat[row * columns + column] = matrix[row][column];
		}
		return flat;
	}

	private static double nanosToMillis(long nanos) {
		return nanos / 1_000_000.0d;
	}

	private static Path writeReport(BackendResult seq, BackendResult mt, double speedup) {
		try {
			Files.createDirectories(REPORT_DIR);
			String json = String.format(Locale.ROOT,
					"""
					{
					  "timestamp": "%s",
					  "benchmark": "turboquant-hat-fused-attention",
					  "headDim": %d,
					  "seqLen": %d,
					  "bits": %d,
					  "warmupRuns": %d,
					  "measuredRuns": %d,
					  "sequential": {
					    "backend": "%s",
					    "avgMillis": %.6f,
					    "maxAbsDiff": %.8f,
					    "meanAbsDiff": %.8f,
					    "cosineSimilarity": %.8f
					  },
					  "multithreaded": {
					    "backend": "%s",
					    "avgMillis": %.6f,
					    "maxAbsDiff": %.8f,
					    "meanAbsDiff": %.8f,
					    "cosineSimilarity": %.8f
					  },
					  "speedupSeqOverMt": %.6f
					}
					""",
					Instant.now(),
					HEAD_DIM,
					SEQ_LEN,
					BITS,
					WARMUP_RUNS,
					MEASURED_RUNS,
					seq.backendClassName(),
					nanosToMillis(seq.avgNanos()),
					seq.comparison().maxAbsDiff(),
					seq.comparison().meanAbsDiff(),
					seq.comparison().cosineSimilarity(),
					mt.backendClassName(),
					nanosToMillis(mt.avgNanos()),
					mt.comparison().maxAbsDiff(),
					mt.comparison().meanAbsDiff(),
					mt.comparison().cosineSimilarity(),
					speedup);
			Files.writeString(REPORT_FILE, json);
			return REPORT_FILE;
		} catch (IOException e) {
			throw new RuntimeException("Failed to write HAT TurboQuant report", e);
		}
	}

	private record BackendResult(String backendClassName, long avgNanos, TurboQuantCore.Comparison comparison) {
	}
}