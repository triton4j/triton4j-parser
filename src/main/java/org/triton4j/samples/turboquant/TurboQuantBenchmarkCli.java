package org.triton4j.samples.turboquant;

public final class TurboQuantBenchmarkCli {

	private TurboQuantBenchmarkCli() {
	}

	public static void main(String[] args) {
		int headDim = 64;
		int seqLen = 128;
		int bits = 4;
		int warmupRuns = 3;
		int measuredRuns = 10;
		long seed = 42L;

		for (int index = 0; index < args.length; index++) {
			String arg = args[index];
			switch (arg) {
			case "--head-dim" -> headDim = Integer.parseInt(args[++index]);
			case "--seq-len" -> seqLen = Integer.parseInt(args[++index]);
			case "--bits" -> bits = Integer.parseInt(args[++index]);
			case "--warmup-runs" -> warmupRuns = Integer.parseInt(args[++index]);
			case "--measured-runs" -> measuredRuns = Integer.parseInt(args[++index]);
			case "--seed" -> seed = Long.parseLong(args[++index]);
			default -> throw new IllegalArgumentException("Unknown argument: " + arg);
			}
		}

		TurboQuantAttentionBenchmark.BenchmarkResult result = TurboQuantAttentionBenchmark.run(
				headDim,
				seqLen,
				bits,
				warmupRuns,
				measuredRuns,
				seed);
		var reportPath = TurboQuantAttentionBenchmark.writeReport(result);

		System.out.printf("TurboQuant benchmark written to %s%n", reportPath);
		System.out.printf("  score timings (ms): fused=%.3f dequantized=%.3f original=%.3f%n",
				result.fusedScoreAvgNanos() / 1_000_000.0d,
				result.dequantizedScoreAvgNanos() / 1_000_000.0d,
				result.originalScoreAvgNanos() / 1_000_000.0d);
		System.out.printf("  attention timings (ms): fused=%.3f dequantized=%.3f original=%.3f%n",
				result.fusedAttentionAvgNanos() / 1_000_000.0d,
				result.dequantizedAttentionAvgNanos() / 1_000_000.0d,
				result.originalAttentionAvgNanos() / 1_000_000.0d);
	}
}