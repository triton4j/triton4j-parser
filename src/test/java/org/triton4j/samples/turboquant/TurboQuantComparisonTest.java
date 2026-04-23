package org.triton4j.samples.turboquant;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Locale;
import java.util.Random;

import org.junit.jupiter.api.Test;

class TurboQuantComparisonTest {

	@Test
	void fusedTurboQuantRemainsCloseToTraditionalJavaAttention() {
		int[][] configs = {
				{ 64, 96, 4 },
				{ 64, 256, 4 },
				{ 128, 256, 3 }
		};

		for (int[] config : configs) {
			int headDim = config[0];
			int seqLen = config[1];
			int bits = config[2];

			TurboQuantCore turboQuant = new TurboQuantCore(headDim, bits, 17L + headDim + bits);
			Random random = new Random(101L + seqLen + bits);
			float[] query = TurboQuantSample.randomVector(random, headDim);
			float[][] keys = TurboQuantSample.randomMatrix(random, seqLen, headDim);
			float[][] values = TurboQuantSample.randomMatrix(random, seqLen, headDim);
			TurboQuantCore.QuantizedVectors quantizedKeys = turboQuant.quantize(keys);
			float[] rotatedQuery = turboQuant.rotate(query);
			float scale = TurboQuantCore.attentionScale(headDim);

			float[] fusedScores = turboQuant.fusedScores(rotatedQuery, quantizedKeys, scale);
			float[] originalScores = turboQuant.referenceScores(query, keys, scale);
			float[] fusedAttention = turboQuant.fusedAttention(rotatedQuery, quantizedKeys, values, scale);
			float[] originalAttention = turboQuant.referenceAttention(query, keys, values, scale);

			TurboQuantCore.Comparison scoreComparison = TurboQuantCore.compare(originalScores, fusedScores);
			TurboQuantCore.Comparison attentionComparison = TurboQuantCore.compare(originalAttention, fusedAttention);

			System.out.printf(Locale.ROOT,
					"TurboQuant compare headDim=%d seqLen=%d bits=%d | score cosine=%.6f maxDiff=%.6f | attention cosine=%.6f maxDiff=%.6f%n",
					headDim,
					seqLen,
					bits,
					scoreComparison.cosineSimilarity(),
					scoreComparison.maxAbsDiff(),
					attentionComparison.cosineSimilarity(),
					attentionComparison.maxAbsDiff());

			assertTrue(scoreComparison.cosineSimilarity() > 0.85f,
					"Expected TurboQuant fused scores to stay close to traditional Java scores.");
			assertTrue(attentionComparison.cosineSimilarity() > 0.85f,
					"Expected TurboQuant fused attention to stay close to traditional Java attention.");
			assertTrue(attentionComparison.maxAbsDiff() < 1.0f,
					"Expected TurboQuant fused attention max diff to remain bounded against traditional Java.");
		}
	}
}
