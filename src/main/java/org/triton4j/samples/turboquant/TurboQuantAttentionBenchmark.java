package org.triton4j.samples.turboquant;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.Locale;
import java.util.Random;

public final class TurboQuantAttentionBenchmark {

	private static final Path REPORT_DIR = Paths.get("build", "reports", "performance");
	private static final Path REPORT_FILE = REPORT_DIR.resolve("turboquant-attention.json");

	private TurboQuantAttentionBenchmark() {
	}

	public static BenchmarkResult run(int headDim, int seqLen, int bits, int warmupRuns, int measuredRuns, long seed) {
		TurboQuantCore turboQuant = new TurboQuantCore(headDim, bits, seed);
		Random random = new Random(seed);
		float[] query = TurboQuantSample.randomVector(random, headDim);
		float[][] keys = TurboQuantSample.randomMatrix(random, seqLen, headDim);
		float[][] values = TurboQuantSample.randomMatrix(random, seqLen, headDim);
		TurboQuantCore.QuantizedVectors quantizedKeys = turboQuant.quantize(keys);
		float[][] dequantizedKeys = turboQuant.dequantize(quantizedKeys);
		float[] rotatedQuery = turboQuant.rotate(query);
		float scale = TurboQuantCore.attentionScale(headDim);

		for (int run = 0; run < warmupRuns; run++) {
			turboQuant.fusedScores(rotatedQuery, quantizedKeys, scale);
			turboQuant.referenceScores(query, dequantizedKeys, scale);
			turboQuant.referenceScores(query, keys, scale);
			turboQuant.fusedAttention(rotatedQuery, quantizedKeys, values, scale);
			turboQuant.dequantizedAttention(query, quantizedKeys, values, scale);
			turboQuant.referenceAttention(query, keys, values, scale);
		}

		long fusedTotalNanos = 0L;
		float[] fusedScores = null;
		for (int run = 0; run < measuredRuns; run++) {
			long start = System.nanoTime();
			fusedScores = turboQuant.fusedScores(rotatedQuery, quantizedKeys, scale);
			fusedTotalNanos += System.nanoTime() - start;
		}

		long dequantizedReferenceTotalNanos = 0L;
		float[] dequantizedReferenceScores = null;
		for (int run = 0; run < measuredRuns; run++) {
			long start = System.nanoTime();
			dequantizedReferenceScores = turboQuant.referenceScores(query, dequantizedKeys, scale);
			dequantizedReferenceTotalNanos += System.nanoTime() - start;
		}

		long originalReferenceTotalNanos = 0L;
		float[] originalReferenceScores = null;
		for (int run = 0; run < measuredRuns; run++) {
			long start = System.nanoTime();
			originalReferenceScores = turboQuant.referenceScores(query, keys, scale);
			originalReferenceTotalNanos += System.nanoTime() - start;
		}

		long fusedAttentionTotalNanos = 0L;
		float[] fusedAttention = null;
		for (int run = 0; run < measuredRuns; run++) {
			long start = System.nanoTime();
			fusedAttention = turboQuant.fusedAttention(rotatedQuery, quantizedKeys, values, scale);
			fusedAttentionTotalNanos += System.nanoTime() - start;
		}

		long dequantizedAttentionTotalNanos = 0L;
		float[] dequantizedAttention = null;
		for (int run = 0; run < measuredRuns; run++) {
			long start = System.nanoTime();
			dequantizedAttention = turboQuant.dequantizedAttention(query, quantizedKeys, values, scale);
			dequantizedAttentionTotalNanos += System.nanoTime() - start;
		}

		long originalAttentionTotalNanos = 0L;
		float[] originalAttention = null;
		for (int run = 0; run < measuredRuns; run++) {
			long start = System.nanoTime();
			originalAttention = turboQuant.referenceAttention(query, keys, values, scale);
			originalAttentionTotalNanos += System.nanoTime() - start;
		}

		TurboQuantCore.Comparison scoreVsDequantized = TurboQuantCore.compare(dequantizedReferenceScores, fusedScores);
		TurboQuantCore.Comparison scoreVsOriginal = TurboQuantCore.compare(originalReferenceScores, fusedScores);
		TurboQuantCore.Comparison attentionVsDequantized = TurboQuantCore.compare(dequantizedAttention, fusedAttention);
		TurboQuantCore.Comparison attentionVsOriginal = TurboQuantCore.compare(originalAttention, fusedAttention);
		return new BenchmarkResult(
				headDim,
				seqLen,
				bits,
				warmupRuns,
				measuredRuns,
				fusedTotalNanos / measuredRuns,
				dequantizedReferenceTotalNanos / measuredRuns,
				originalReferenceTotalNanos / measuredRuns,
				fusedAttentionTotalNanos / measuredRuns,
				dequantizedAttentionTotalNanos / measuredRuns,
				originalAttentionTotalNanos / measuredRuns,
				scoreVsDequantized,
				scoreVsOriginal,
				attentionVsDequantized,
				attentionVsOriginal,
				TurboQuantCore.theoreticalCompressedSizeBytes(seqLen, headDim, bits),
				seqLen * headDim * Float.BYTES);
	}

	public static Path writeReport(BenchmarkResult result) {
		try {
			Files.createDirectories(REPORT_DIR);
			String json = String.format(Locale.ROOT,
					"""
					{
					  \"timestamp\": \"%s\",
					  \"benchmark\": \"turboquant-attention\",
					  \"headDim\": %d,
					  \"seqLen\": %d,
					  \"bits\": %d,
					  \"warmupRuns\": %d,
					  \"measuredRuns\": %d,
					  \"scoreTimings\": {
					    \"fusedAvgMillis\": %.6f,
					    \"dequantizedReferenceAvgMillis\": %.6f,
					    \"originalReferenceAvgMillis\": %.6f
					  },
					  \"attentionTimings\": {
					    \"fusedAvgMillis\": %.6f,
					    \"dequantizedReferenceAvgMillis\": %.6f,
					    \"originalReferenceAvgMillis\": %.6f
					  },
					  \"scoreVsDequantized\": {
					    \"maxAbsDiff\": %.8f,
					    \"meanAbsDiff\": %.8f,
					    \"cosineSimilarity\": %.8f
					  },
					  \"scoreVsOriginal\": {
					    \"maxAbsDiff\": %.8f,
					    \"meanAbsDiff\": %.8f,
					    \"cosineSimilarity\": %.8f
					  },
					  \"attentionVsDequantized\": {
					    \"maxAbsDiff\": %.8f,
					    \"meanAbsDiff\": %.8f,
					    \"cosineSimilarity\": %.8f
					  },
					  \"attentionVsOriginal\": {
					    \"maxAbsDiff\": %.8f,
					    \"meanAbsDiff\": %.8f,
					    \"cosineSimilarity\": %.8f
					  },
					  \"originalBytes\": %d,
					  \"compressedBytes\": %d,
					  \"compressionRatio\": %.6f
					}
					""",
					Instant.now(),
					result.headDim(),
					result.seqLen(),
					result.bits(),
					result.warmupRuns(),
					result.measuredRuns(),
					result.fusedScoreAvgNanos() / 1_000_000.0d,
					result.dequantizedScoreAvgNanos() / 1_000_000.0d,
					result.originalScoreAvgNanos() / 1_000_000.0d,
					result.fusedAttentionAvgNanos() / 1_000_000.0d,
					result.dequantizedAttentionAvgNanos() / 1_000_000.0d,
					result.originalAttentionAvgNanos() / 1_000_000.0d,
					result.scoreVsDequantized().maxAbsDiff(),
					result.scoreVsDequantized().meanAbsDiff(),
					result.scoreVsDequantized().cosineSimilarity(),
					result.scoreVsOriginal().maxAbsDiff(),
					result.scoreVsOriginal().meanAbsDiff(),
					result.scoreVsOriginal().cosineSimilarity(),
					result.attentionVsDequantized().maxAbsDiff(),
					result.attentionVsDequantized().meanAbsDiff(),
					result.attentionVsDequantized().cosineSimilarity(),
					result.attentionVsOriginal().maxAbsDiff(),
					result.attentionVsOriginal().meanAbsDiff(),
					result.attentionVsOriginal().cosineSimilarity(),
					result.originalBytes(),
					result.compressedBytes(),
					(double) result.originalBytes() / Math.max(result.compressedBytes(), 1));
			Files.writeString(REPORT_FILE, json);
			return REPORT_FILE;
		} catch (IOException e) {
			throw new RuntimeException("Failed to write TurboQuant report", e);
		}
	}

	public record BenchmarkResult(
			int headDim,
			int seqLen,
			int bits,
			int warmupRuns,
			int measuredRuns,
			long fusedScoreAvgNanos,
			long dequantizedScoreAvgNanos,
			long originalScoreAvgNanos,
			long fusedAttentionAvgNanos,
			long dequantizedAttentionAvgNanos,
			long originalAttentionAvgNanos,
			TurboQuantCore.Comparison scoreVsDequantized,
			TurboQuantCore.Comparison scoreVsOriginal,
			TurboQuantCore.Comparison attentionVsDequantized,
			TurboQuantCore.Comparison attentionVsOriginal,
			int compressedBytes,
			int originalBytes) {
	}
}