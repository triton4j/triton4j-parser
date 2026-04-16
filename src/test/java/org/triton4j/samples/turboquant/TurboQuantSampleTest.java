package org.triton4j.samples.turboquant;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;

import org.junit.jupiter.api.Test;

class TurboQuantSampleTest {

	@Test
	void fusedReferenceMatchesDequantizedScores() {
		int headDim = 64;
		int seqLen = 48;
		TurboQuantCore turboQuant = new TurboQuantCore(headDim, 4, 7L);
		Random random = new Random(11L);

		float[] query = TurboQuantSample.randomVector(random, headDim);
		float[][] keys = TurboQuantSample.randomMatrix(random, seqLen, headDim);
		TurboQuantCore.QuantizedVectors quantizedKeys = turboQuant.quantize(keys);
		TurboQuantCore.Comparison comparison = turboQuant.compareFusedAgainstDequantized(
				query,
				quantizedKeys,
				TurboQuantCore.attentionScale(headDim));

		assertTrue(comparison.maxAbsDiff() < 1.0e-3f,
				"Expected fused path to match dequantized scores, got max diff=" + comparison.maxAbsDiff());
		assertTrue(comparison.cosineSimilarity() > 0.9999f,
				"Expected fused path to preserve score direction, got cosine=" + comparison.cosineSimilarity());
	}
}