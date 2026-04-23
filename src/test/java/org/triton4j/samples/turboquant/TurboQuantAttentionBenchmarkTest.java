package org.triton4j.samples.turboquant;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

import org.junit.jupiter.api.Test;

class TurboQuantAttentionBenchmarkTest {

	@Test
	void writesBenchmarkReportAndKeepsScoresAligned() {
		TurboQuantAttentionBenchmark.BenchmarkResult result = TurboQuantAttentionBenchmark.run(64, 96, 4, 1, 4, 19L);
		Path report = TurboQuantAttentionBenchmark.writeReport(result);

		assertTrue(result.fusedScoreAvgNanos() > 0L, "Expected fused score timing to be positive.");
		assertTrue(result.dequantizedScoreAvgNanos() > 0L, "Expected dequantized score timing to be positive.");
		assertTrue(result.originalScoreAvgNanos() > 0L, "Expected original score timing to be positive.");
		assertTrue(result.fusedAttentionAvgNanos() > 0L, "Expected fused attention timing to be positive.");
		assertTrue(result.dequantizedAttentionAvgNanos() > 0L, "Expected dequantized attention timing to be positive.");
		assertTrue(result.originalAttentionAvgNanos() > 0L, "Expected original attention timing to be positive.");
		assertTrue(result.scoreVsDequantized().maxAbsDiff() < 1.0e-3f,
				"Expected fused scores to match dequantized reference, got max diff=" + result.scoreVsDequantized().maxAbsDiff());
		assertTrue(result.scoreVsDequantized().cosineSimilarity() > 0.9999f,
				"Expected fused score cosine similarity > 0.9999, got " + result.scoreVsDequantized().cosineSimilarity());
		assertTrue(result.scoreVsOriginal().cosineSimilarity() > 0.90f,
				"Expected fused scores to remain directionally close to original keys, got " + result.scoreVsOriginal().cosineSimilarity());
		assertTrue(result.attentionVsDequantized().maxAbsDiff() < 1.0e-3f,
				"Expected fused attention output to match dequantized attention, got max diff=" + result.attentionVsDequantized().maxAbsDiff());
		assertTrue(result.attentionVsOriginal().cosineSimilarity() > 0.90f,
				"Expected fused attention output to remain directionally close to original attention, got " + result.attentionVsOriginal().cosineSimilarity());
		assertTrue(Files.exists(report), "Expected benchmark report to exist at " + report.toAbsolutePath());
	}

	@Test
	void reportsEfficiencyAgainstTraditionalJavaAcrossConfigurations() {
		int[][] configs = {
				{ 64, 128, 4 },
				{ 64, 512, 4 },
				{ 128, 512, 3 }
		};

		for (int[] config : configs) {
			int headDim = config[0];
			int seqLen = config[1];
			int bits = config[2];

			TurboQuantAttentionBenchmark.BenchmarkResult result = TurboQuantAttentionBenchmark.run(
					headDim,
					seqLen,
					bits,
					2,
					8,
					31L + headDim + seqLen + bits);

			double scoreSpeedup = (double) result.originalScoreAvgNanos()
					/ Math.max((double) result.fusedScoreAvgNanos(), 1.0d);
			double attentionSpeedup = (double) result.originalAttentionAvgNanos()
					/ Math.max((double) result.fusedAttentionAvgNanos(), 1.0d);
			double compressionRatio = (double) result.originalBytes()
					/ Math.max((double) result.compressedBytes(), 1.0d);

			System.out.printf(Locale.ROOT,
					"TurboQuant efficiency headDim=%d seqLen=%d bits=%d | fusedAttention=%.3f ms originalAttention=%.3f ms speedup=%.3fx compression=%.3fx cosine=%.6f%n",
					headDim,
					seqLen,
					bits,
					nanosToMillis(result.fusedAttentionAvgNanos()),
					nanosToMillis(result.originalAttentionAvgNanos()),
					attentionSpeedup,
					compressionRatio,
					result.attentionVsOriginal().cosineSimilarity());

			assertTrue(result.fusedScoreAvgNanos() > 0L, "Expected fused score timing to be positive.");
			assertTrue(result.originalScoreAvgNanos() > 0L, "Expected original score timing to be positive.");
			assertTrue(result.fusedAttentionAvgNanos() > 0L, "Expected fused attention timing to be positive.");
			assertTrue(result.originalAttentionAvgNanos() > 0L, "Expected original attention timing to be positive.");
			assertTrue(scoreSpeedup > 0.0d, "Expected score speedup ratio to be positive.");
			assertTrue(attentionSpeedup > 0.0d, "Expected attention speedup ratio to be positive.");
			assertTrue(compressionRatio > 1.0d,
					"Expected TurboQuant compressed representation to be smaller than original Java baseline.");
			assertTrue(result.scoreVsOriginal().cosineSimilarity() > 0.85f,
					"Expected fused scores to stay reasonably aligned with original Java scores.");
			assertTrue(result.attentionVsOriginal().cosineSimilarity() > 0.85f,
					"Expected fused attention to stay reasonably aligned with original Java attention.");
		}
	}

	private static double nanosToMillis(long nanos) {
		return nanos / 1_000_000.0d;
	}
}
