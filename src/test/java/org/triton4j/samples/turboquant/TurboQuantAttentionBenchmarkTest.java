package org.triton4j.samples.turboquant;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;

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
}