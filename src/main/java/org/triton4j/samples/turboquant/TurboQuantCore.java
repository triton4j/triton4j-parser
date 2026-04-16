package org.triton4j.samples.turboquant;

import java.util.Arrays;
import java.util.Random;

public final class TurboQuantCore {

	private static final double MIN_NORM = 1e-8d;
	private static final int DEFAULT_LLOYD_ITERATIONS = 80;
	private static final int DEFAULT_GRID_SIZE = 8192;

	private final int dimension;
	private final int bits;
	private final float[][] rotation;
	private final float[][] rotationTranspose;
	private final float[] centroids;
	private final float[] boundaries;

	public TurboQuantCore(int dimension, int bits) {
		this(dimension, bits, 0L);
	}

	public TurboQuantCore(int dimension, int bits, long rotationSeed) {
		if (dimension <= 0)
			throw new IllegalArgumentException("dimension must be positive");
		if (bits <= 0 || bits > 8)
			throw new IllegalArgumentException("bits must be in the range [1, 8]");
		this.dimension = dimension;
		this.bits = bits;
		this.rotation = makeRotationMatrix(dimension, rotationSeed);
		this.rotationTranspose = transpose(this.rotation);
		this.centroids = buildLloydMaxCodebook(dimension, bits, DEFAULT_LLOYD_ITERATIONS, DEFAULT_GRID_SIZE);
		this.boundaries = buildBoundaries(this.centroids);
	}

	public int dimension() {
		return dimension;
	}

	public int bits() {
		return bits;
	}

	public float[] centroids() {
		return Arrays.copyOf(centroids, centroids.length);
	}

	public float[] rotate(float[] vector) {
		validateVector(vector);
		return matVec(vector, rotationTranspose);
	}

	public QuantizedVectors quantize(float[][] vectors) {
		validateMatrix(vectors);
		int[][] indices = new int[vectors.length][dimension];
		float[] norms = new float[vectors.length];
		for (int row = 0; row < vectors.length; row++) {
			float[] vector = vectors[row];
			double norm = l2Norm(vector);
			norms[row] = (float) norm;
			float invNorm = (float) (1.0d / norm);
			float[] normalized = new float[dimension];
			for (int column = 0; column < dimension; column++)
				normalized[column] = vector[column] * invNorm;
			float[] rotated = matVec(normalized, rotationTranspose);
			for (int column = 0; column < dimension; column++)
				indices[row][column] = searchSorted(boundaries, rotated[column]);
		}
		return new QuantizedVectors(indices, norms);
	}

	public float[][] dequantize(QuantizedVectors quantizedVectors) {
		validateQuantizedVectors(quantizedVectors);
		int[][] indices = quantizedVectors.indices();
		float[] norms = quantizedVectors.norms();
		float[][] reconstructed = new float[indices.length][dimension];
		for (int row = 0; row < indices.length; row++) {
			float[] rotated = new float[dimension];
			for (int column = 0; column < dimension; column++)
				rotated[column] = centroids[indices[row][column]];
			float[] unit = matVec(rotated, rotation);
			for (int column = 0; column < dimension; column++)
				reconstructed[row][column] = unit[column] * norms[row];
		}
		return reconstructed;
	}

	public float[] fusedScores(float[] rotatedQuery, QuantizedVectors quantizedVectors, float scale) {
		validateVector(rotatedQuery);
		validateQuantizedVectors(quantizedVectors);
		int[][] indices = quantizedVectors.indices();
		float[] norms = quantizedVectors.norms();
		float[] scores = new float[indices.length];
		for (int row = 0; row < indices.length; row++) {
			float dot = 0.0f;
			for (int column = 0; column < dimension; column++)
				dot += rotatedQuery[column] * centroids[indices[row][column]];
			scores[row] = norms[row] * dot * scale;
		}
		return scores;
	}

	public float[] referenceScores(float[] query, float[][] keys, float scale) {
		validateVector(query);
		validateMatrix(keys);
		float[] scores = new float[keys.length];
		for (int row = 0; row < keys.length; row++) {
			float dot = 0.0f;
			for (int column = 0; column < dimension; column++)
				dot += query[column] * keys[row][column];
			scores[row] = dot * scale;
		}
		return scores;
	}

	public float[] attentionOutput(float[] scores, float[][] values) {
		validateScores(scores);
		validateMatrix(values);
		if (scores.length != values.length)
			throw new IllegalArgumentException("scores length must match number of value rows");
		float[] weights = softmax(scores);
		float[] output = new float[dimension];
		for (int row = 0; row < values.length; row++) {
			float weight = weights[row];
			for (int column = 0; column < dimension; column++)
				output[column] += weight * values[row][column];
		}
		return output;
	}

	public float[] referenceAttention(float[] query, float[][] keys, float[][] values, float scale) {
		return attentionOutput(referenceScores(query, keys, scale), values);
	}

	public float[] fusedAttention(float[] rotatedQuery, QuantizedVectors quantizedKeys, float[][] values, float scale) {
		return attentionOutput(fusedScores(rotatedQuery, quantizedKeys, scale), values);
	}

	public float[] dequantizedAttention(float[] query, QuantizedVectors quantizedKeys, float[][] values, float scale) {
		return attentionOutput(referenceScores(query, dequantize(quantizedKeys), scale), values);
	}

	public Comparison compareFusedAgainstDequantized(float[] query, QuantizedVectors quantizedVectors, float scale) {
		float[] fused = fusedScores(rotate(query), quantizedVectors, scale);
		float[] dequantizedReference = referenceScores(query, dequantize(quantizedVectors), scale);
		return Comparison.from(dequantizedReference, fused);
	}

	public static Comparison compare(float[] reference, float[] actual) {
		return Comparison.from(reference, actual);
	}

	public static float attentionScale(int headDim) {
		return (float) (1.0d / Math.sqrt(headDim));
	}

	public static int theoreticalCompressedSizeBytes(int vectorCount, int dimension, int bits) {
		int indexBytes = (vectorCount * dimension * bits + 7) / 8;
		int normBytes = vectorCount * 2;
		return indexBytes + normBytes;
	}

	private void validateVector(float[] vector) {
		if (vector == null || vector.length != dimension)
			throw new IllegalArgumentException("Expected a vector of length " + dimension);
	}

	private void validateMatrix(float[][] matrix) {
		if (matrix == null)
			throw new IllegalArgumentException("matrix must not be null");
		for (float[] row : matrix)
			validateVector(row);
	}

	private void validateScores(float[] scores) {
		if (scores == null)
			throw new IllegalArgumentException("scores must not be null");
	}

	private void validateQuantizedVectors(QuantizedVectors quantizedVectors) {
		if (quantizedVectors == null)
			throw new IllegalArgumentException("quantizedVectors must not be null");
		if (quantizedVectors.indices().length != quantizedVectors.norms().length)
			throw new IllegalArgumentException("indices and norms must have the same number of rows");
		for (int[] row : quantizedVectors.indices()) {
			if (row == null || row.length != dimension)
				throw new IllegalArgumentException("Each quantized row must have dimension " + dimension);
		}
	}

	private static double l2Norm(float[] vector) {
		double sum = 0.0d;
		for (float value : vector)
			sum += value * value;
		return Math.max(Math.sqrt(sum), MIN_NORM);
	}

	private static int searchSorted(float[] sortedBoundaries, float value) {
		int low = 0;
		int high = sortedBoundaries.length;
		while (low < high) {
			int mid = (low + high) >>> 1;
			if (value > sortedBoundaries[mid])
				low = mid + 1;
			else
				high = mid;
		}
		return low;
	}

	private static float[] matVec(float[] vector, float[][] matrix) {
		int size = vector.length;
		float[] out = new float[size];
		for (int column = 0; column < size; column++) {
			float sum = 0.0f;
			for (int row = 0; row < size; row++)
				sum += vector[row] * matrix[row][column];
			out[column] = sum;
		}
		return out;
	}

	private static float[][] transpose(float[][] matrix) {
		int size = matrix.length;
		float[][] transpose = new float[size][size];
		for (int row = 0; row < size; row++) {
			for (int column = 0; column < size; column++)
				transpose[column][row] = matrix[row][column];
		}
		return transpose;
	}

	private static float[] buildBoundaries(float[] centroids) {
		float[] boundaries = new float[centroids.length - 1];
		for (int index = 0; index < boundaries.length; index++)
			boundaries[index] = (centroids[index] + centroids[index + 1]) * 0.5f;
		return boundaries;
	}

	private static float[] buildLloydMaxCodebook(int dimension, int bits, int iterations, int gridSize) {
		int nLevels = 1 << bits;
		double sigma = dimension > 1 ? 1.0d / Math.sqrt(dimension) : 0.5d;
		double lo = Math.max(-1.0d + 1e-7d, -6.0d * sigma);
		double hi = Math.min(1.0d - 1e-7d, 6.0d * sigma);
		double[] grid = new double[gridSize];
		double[] pdf = new double[gridSize];
		double step = (hi - lo) / (gridSize - 1);
		double pdfSum = 0.0d;
		for (int index = 0; index < gridSize; index++) {
			double x = lo + step * index;
			grid[index] = x;
			pdf[index] = betaPdfUnnormalized(x, dimension);
			pdfSum += pdf[index];
		}
		for (int index = 0; index < gridSize; index++)
			pdf[index] /= pdfSum;

		double[] cdf = new double[gridSize];
		double running = 0.0d;
		for (int index = 0; index < gridSize; index++) {
			running += pdf[index];
			cdf[index] = running;
		}

		double[] centroids = new double[nLevels];
		for (int level = 0; level < nLevels; level++) {
			double target = (level + 0.5d) / nLevels;
			int centroidIndex = 0;
			while (centroidIndex < gridSize - 1 && cdf[centroidIndex] < target)
				centroidIndex++;
			centroids[level] = grid[centroidIndex];
		}

		int[] assignments = new int[gridSize];
		for (int iteration = 0; iteration < iterations; iteration++) {
			for (int point = 0; point < gridSize; point++) {
				double bestDistance = Double.POSITIVE_INFINITY;
				int bestLevel = 0;
				for (int level = 0; level < nLevels; level++) {
					double distance = Math.abs(grid[point] - centroids[level]);
					if (distance < bestDistance) {
						bestDistance = distance;
						bestLevel = level;
					}
				}
				assignments[point] = bestLevel;
			}

			double[] nextCentroids = Arrays.copyOf(centroids, centroids.length);
			double[] weights = new double[nLevels];
			Arrays.fill(nextCentroids, 0.0d);
			for (int point = 0; point < gridSize; point++) {
				int level = assignments[point];
				nextCentroids[level] += grid[point] * pdf[point];
				weights[level] += pdf[point];
			}
			for (int level = 0; level < nLevels; level++) {
				if (weights[level] > 0.0d)
					nextCentroids[level] /= weights[level];
				else
					nextCentroids[level] = centroids[level];
			}
			centroids = nextCentroids;
		}

		Arrays.sort(centroids);
		float[] result = new float[centroids.length];
		for (int index = 0; index < centroids.length; index++)
			result[index] = (float) centroids[index];
		return result;
	}

	private static double betaPdfUnnormalized(double x, int dimension) {
		double alpha = (dimension - 1.0d) / 2.0d;
		double base = Math.max(1.0d - (x * x), 1e-30d);
		return Math.exp((alpha - 1.0d) * Math.log(base));
	}

	private static float[][] makeRotationMatrix(int dimension, long seed) {
		Random random = new Random(seed);
		float[][] columns = new float[dimension][dimension];
		for (int column = 0; column < dimension; column++) {
			float[] candidate = new float[dimension];
			boolean accepted = false;
			while (!accepted) {
				for (int row = 0; row < dimension; row++)
					candidate[row] = (float) random.nextGaussian();
				for (int prev = 0; prev < column; prev++) {
					float projection = dot(candidate, columns[prev]);
					for (int row = 0; row < dimension; row++)
						candidate[row] -= projection * columns[prev][row];
				}
				double norm = l2Norm(candidate);
				if (norm > 1e-5d) {
					float invNorm = (float) (1.0d / norm);
					for (int row = 0; row < dimension; row++)
						columns[column][row] = candidate[row] * invNorm;
					accepted = true;
				}
			}
		}

		float[][] matrix = new float[dimension][dimension];
		for (int row = 0; row < dimension; row++) {
			for (int column = 0; column < dimension; column++)
				matrix[row][column] = columns[column][row];
		}
		return matrix;
	}

	private static float dot(float[] left, float[] right) {
		float sum = 0.0f;
		for (int index = 0; index < left.length; index++)
			sum += left[index] * right[index];
		return sum;
	}

	static float[] softmax(float[] logits) {
		if (logits.length == 0)
			return new float[0];
		float max = logits[0];
		for (int index = 1; index < logits.length; index++)
			max = Math.max(max, logits[index]);
		float[] weights = new float[logits.length];
		double sum = 0.0d;
		for (int index = 0; index < logits.length; index++) {
			weights[index] = (float) Math.exp(logits[index] - max);
			sum += weights[index];
		}
		float inv = (float) (1.0d / Math.max(sum, MIN_NORM));
		for (int index = 0; index < weights.length; index++)
			weights[index] *= inv;
		return weights;
	}

	public record QuantizedVectors(int[][] indices, float[] norms) {
	}

	public record Comparison(float maxAbsDiff, float meanAbsDiff, float cosineSimilarity) {

		private static Comparison from(float[] reference, float[] actual) {
			if (reference.length != actual.length)
				throw new IllegalArgumentException("reference and actual arrays must have the same length");
			float max = 0.0f;
			float sum = 0.0f;
			double dot = 0.0d;
			double refNorm = 0.0d;
			double actualNorm = 0.0d;
			for (int index = 0; index < reference.length; index++) {
				float diff = Math.abs(reference[index] - actual[index]);
				max = Math.max(max, diff);
				sum += diff;
				dot += reference[index] * actual[index];
				refNorm += reference[index] * reference[index];
				actualNorm += actual[index] * actual[index];
			}
			float cosine = (float) (dot / Math.max(Math.sqrt(refNorm) * Math.sqrt(actualNorm), MIN_NORM));
			return new Comparison(max, sum / reference.length, cosine);
		}
	}
}