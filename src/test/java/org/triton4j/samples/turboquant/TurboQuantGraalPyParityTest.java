package org.triton4j.samples.turboquant;

import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.lang.reflect.Method;
import java.util.Locale;
import java.util.Random;

import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.jupiter.api.Test;

import jdk.incubator.code.Reflect;

class TurboQuantGraalPyParityTest {

	private static final float FLOAT_TOLERANCE = 1.0e-5f;

	@Test
	void graalPyReferenceMatchesTurboQuantFusedMath() throws Exception {
		int headDim = 8;
		int seqLen = 6;
		int bits = 3;
		long rotationSeed = 17L;
		long dataSeed = 29L;

		TurboQuantCore turboQuant = new TurboQuantCore(headDim, bits, rotationSeed);
		Random random = new Random(dataSeed);
		float[] query = TurboQuantSample.randomVector(random, headDim);
		float[][] keys = TurboQuantSample.randomMatrix(random, seqLen, headDim);
		float[][] values = TurboQuantSample.randomMatrix(random, seqLen, headDim);
		TurboQuantCore.QuantizedVectors quantizedKeys = turboQuant.quantize(keys);
		float[] rotatedQuery = turboQuant.rotate(query);
		float[] centroids = turboQuant.centroids();
		float[] keyNorms = quantizedKeys.norms();
		int[][] keyIndices = quantizedKeys.indices();
		float scale = TurboQuantCore.attentionScale(headDim);

		float[] javaScores = turboQuant.fusedScores(rotatedQuery, quantizedKeys, scale);
		float[] javaAttention = turboQuant.fusedAttention(rotatedQuery, quantizedKeys, values, scale);

		try (Context polyglot = Context.newBuilder("python").allowAllAccess(true).build()) {
			polyglot.eval("python", buildPythonSource(rotatedQuery, keyIndices, keyNorms, centroids, values, scale));

			Value pythonBindings = polyglot.getBindings("python");
			float[] pythonScores = toFloatArray(pythonBindings.getMember("PY_SCORES"));
			float[] pythonAttention = toFloatArray(pythonBindings.getMember("PY_ATTENTION"));

			assertFloatArrayClose(javaScores, pythonScores, FLOAT_TOLERANCE, "fused scores");
			assertFloatArrayClose(javaAttention, pythonAttention, FLOAT_TOLERANCE, "fused attention");
		}

		Method fusedScoresKernel = TurboQuantTritonKernels.class.getDeclaredMethod(
				"fusedQkScoresKernel",
				oracle.code.triton.Ptr.class,
				oracle.code.triton.Ptr.class,
				oracle.code.triton.Ptr.class,
				oracle.code.triton.Ptr.class,
				oracle.code.triton.Ptr.class,
				int.class,
				float.class,
				int.class,
				int.class);
		Method fusedAttentionKernel = TurboQuantTritonKernels.class.getDeclaredMethod(
				"fusedAttentionKernel",
				oracle.code.triton.Ptr.class,
				oracle.code.triton.Ptr.class,
				oracle.code.triton.Ptr.class,
				oracle.code.triton.Ptr.class,
				oracle.code.triton.Ptr.class,
				oracle.code.triton.Ptr.class,
				int.class,
				float.class,
				int.class,
				int.class);

		assertNotNull(fusedScoresKernel.getAnnotation(Reflect.class),
				"Expected fusedQkScoresKernel to remain a reflected Triton entry point.");
		assertNotNull(fusedAttentionKernel.getAnnotation(Reflect.class),
				"Expected fusedAttentionKernel to remain a reflected Triton entry point.");
	}

	private static String buildPythonSource(float[] rotatedQuery, int[][] keyIndices, float[] keyNorms, float[] centroids,
			float[][] values, float scale) {
		return """
				import math

				def fused_scores(rotated_query, key_indices, key_norms, centroids, scale):
				    scores = []
				    for row_index in range(len(key_indices)):
				        dot = 0.0
				        row = key_indices[row_index]
				        for column in range(len(rotated_query)):
				            dot += float(rotated_query[column]) * float(centroids[int(row[column])])
				        scores.append(float(key_norms[row_index]) * dot * float(scale))
				    return scores

				def softmax(scores):
				    max_score = max(scores)
				    exps = [math.exp(score - max_score) for score in scores]
				    normalizer = sum(exps)
				    if normalizer == 0.0:
				        return [0.0 for _ in exps]
				    return [value / normalizer for value in exps]

				def fused_attention(rotated_query, key_indices, key_norms, centroids, values, scale):
				    scores = fused_scores(rotated_query, key_indices, key_norms, centroids, scale)
				    weights = softmax(scores)
				    out = [0.0] * len(values[0])
				    for row_index in range(len(values)):
				        weight = weights[row_index]
				        row = values[row_index]
				        for column in range(len(out)):
				            out[column] += weight * float(row[column])
				    return out

				ROTATED_QUERY = %s
				KEY_INDICES = %s
				KEY_NORMS = %s
				CENTROIDS = %s
				VALUES = %s
				SCALE = %s

				PY_SCORES = fused_scores(ROTATED_QUERY, KEY_INDICES, KEY_NORMS, CENTROIDS, SCALE)
				PY_ATTENTION = fused_attention(ROTATED_QUERY, KEY_INDICES, KEY_NORMS, CENTROIDS, VALUES, SCALE)
				"""
				.formatted(
						toPythonLiteral(rotatedQuery),
						toPythonLiteral(keyIndices),
						toPythonLiteral(keyNorms),
						toPythonLiteral(centroids),
						toPythonLiteral(values),
						formatFloat(scale));
	}

	private static float[] toFloatArray(Value value) {
		int size = Math.toIntExact(value.getArraySize());
		float[] out = new float[size];
		for (int index = 0; index < size; index++)
			out[index] = (float) value.getArrayElement(index).asDouble();
		return out;
	}

	private static void assertFloatArrayClose(float[] expected, float[] actual, float tolerance, String label) {
		org.junit.jupiter.api.Assertions.assertEquals(expected.length, actual.length,
				"Length mismatch for " + label);
		for (int index = 0; index < expected.length; index++) {
			org.junit.jupiter.api.Assertions.assertEquals(expected[index], actual[index], tolerance,
					label + " mismatch at index " + index);
		}
	}

	private static String toPythonLiteral(float[] values) {
		StringBuilder builder = new StringBuilder();
		builder.append('[');
		for (int index = 0; index < values.length; index++) {
			if (index > 0)
				builder.append(", ");
			builder.append(formatFloat(values[index]));
		}
		builder.append(']');
		return builder.toString();
	}

	private static String toPythonLiteral(float[][] values) {
		StringBuilder builder = new StringBuilder();
		builder.append('[');
		for (int index = 0; index < values.length; index++) {
			if (index > 0)
				builder.append(", ");
			builder.append(toPythonLiteral(values[index]));
		}
		builder.append(']');
		return builder.toString();
	}

	private static String toPythonLiteral(int[][] values) {
		StringBuilder builder = new StringBuilder();
		builder.append('[');
		for (int index = 0; index < values.length; index++) {
			if (index > 0)
				builder.append(", ");
			builder.append('[');
			for (int column = 0; column < values[index].length; column++) {
				if (column > 0)
					builder.append(", ");
				builder.append(values[index][column]);
			}
			builder.append(']');
		}
		builder.append(']');
		return builder.toString();
	}

	private static String formatFloat(float value) {
		if (Float.isFinite(value))
			return String.format(Locale.ROOT, "%.9f", value);
		if (Float.isNaN(value))
			return "float('nan')";
		return value > 0.0f ? "float('inf')" : "float('-inf')";
	}
}
