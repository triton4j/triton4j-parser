package org.triton4j.samples.turboquant;

import java.util.Random;

public final class TurboQuantSample {

	private TurboQuantSample() {
	}

	public static void main(String[] args) {
		SampleConfig config = SampleConfig.parse(args);
		TurboQuantCore turboQuant = new TurboQuantCore(config.headDim(), config.bits(), config.seed());
		Random random = new Random(config.seed());

		float[] query = randomVector(random, config.headDim());
		float[][] keys = randomMatrix(random, config.seqLen(), config.headDim());
		TurboQuantCore.QuantizedVectors quantizedKeys = turboQuant.quantize(keys);
		TurboQuantCore.Comparison comparison = turboQuant.compareFusedAgainstDequantized(
				query,
				quantizedKeys,
				TurboQuantCore.attentionScale(config.headDim()));

		int originalBytes = config.seqLen() * config.headDim() * Float.BYTES;
		int compressedBytes = TurboQuantCore.theoreticalCompressedSizeBytes(
				config.seqLen(),
				config.headDim(),
				config.bits());

		System.out.printf("TurboQuant Java sample%n");
		System.out.printf("  headDim=%d  seqLen=%d  bits=%d%n",
				config.headDim(), config.seqLen(), config.bits());
		System.out.printf("  fused vs dequantized max diff:  %.6f%n", comparison.maxAbsDiff());
		System.out.printf("  fused vs dequantized mean diff: %.6f%n", comparison.meanAbsDiff());
		System.out.printf("  fused vs dequantized cosine:    %.6f%n", comparison.cosineSimilarity());
		System.out.printf("  theoretical size: %d -> %d bytes (%.2fx)%n",
				originalBytes,
				compressedBytes,
				(double) originalBytes / Math.max(compressedBytes, 1));

		TurboQuantAttentionBenchmark.BenchmarkResult benchmark = TurboQuantAttentionBenchmark.run(
				config.headDim(),
				config.seqLen(),
				config.bits(),
				config.warmupRuns(),
				config.measuredRuns(),
				config.seed());
		var reportPath = TurboQuantAttentionBenchmark.writeReport(benchmark);
		System.out.printf("  fused avg: %.3f ms  dequantized avg: %.3f ms%n",
				benchmark.fusedScoreAvgNanos() / 1_000_000.0d,
				benchmark.dequantizedScoreAvgNanos() / 1_000_000.0d);
		System.out.printf("  original-key score avg: %.3f ms%n",
				benchmark.originalScoreAvgNanos() / 1_000_000.0d);
		System.out.printf("  attention avg (fused/dequantized/original): %.3f / %.3f / %.3f ms%n",
				benchmark.fusedAttentionAvgNanos() / 1_000_000.0d,
				benchmark.dequantizedAttentionAvgNanos() / 1_000_000.0d,
				benchmark.originalAttentionAvgNanos() / 1_000_000.0d);
		System.out.printf("  score cosine vs original: %.6f  attention cosine vs original: %.6f%n",
				benchmark.scoreVsOriginal().cosineSimilarity(),
				benchmark.attentionVsOriginal().cosineSimilarity());
		System.out.printf("  benchmark report: %s%n", reportPath);
		System.out.println();
		System.out.println("Triton kernel entry points: org.triton4j.samples.turboquant.TurboQuantTritonKernels#fusedQkScoresKernel and #fusedAttentionKernel");
	}

	static float[] randomVector(Random random, int size) {
		float[] vector = new float[size];
		for (int index = 0; index < size; index++)
			vector[index] = (float) random.nextGaussian();
		return vector;
	}

	static float[][] randomMatrix(Random random, int rows, int columns) {
		float[][] matrix = new float[rows][columns];
		for (int row = 0; row < rows; row++)
			matrix[row] = randomVector(random, columns);
		return matrix;
	}

	private record SampleConfig(int headDim, int seqLen, int bits, long seed, int warmupRuns, int measuredRuns) {

		private static SampleConfig parse(String[] args) {
			int headDim = 64;
			int seqLen = 128;
			int bits = 4;
			long seed = 42L;
			int warmupRuns = 3;
			int measuredRuns = 10;
			for (int index = 0; index < args.length; index++) {
				String arg = args[index];
				switch (arg) {
				case "--head-dim" -> headDim = Integer.parseInt(args[++index]);
				case "--seq-len" -> seqLen = Integer.parseInt(args[++index]);
				case "--bits" -> bits = Integer.parseInt(args[++index]);
				case "--seed" -> seed = Long.parseLong(args[++index]);
				case "--warmup-runs" -> warmupRuns = Integer.parseInt(args[++index]);
				case "--measured-runs" -> measuredRuns = Integer.parseInt(args[++index]);
				default -> throw new IllegalArgumentException("Unknown argument: " + arg);
				}
			}
			return new SampleConfig(headDim, seqLen, bits, seed, warmupRuns, measuredRuns);
		}
	}
}