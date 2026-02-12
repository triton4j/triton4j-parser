package org.triton4j.codegen.test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.Locale;

import org.junit.jupiter.api.Test;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.S32Array;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface.RO;
import optkl.ifacemapper.MappableIface.WO;

public class HatKernelExecutionTest {

	private static final String JAVA_SEQ_BACKEND = "hat.backend.java.JavaSequentialBackend";
	private static final String JAVA_MT_BACKEND = "hat.backend.java.JavaMultiThreadedBackend";

	private static final int VECTOR_LENGTH = 256 * 1024;
	private static final int WARMUP_RUNS = 2;
	private static final int MEASURED_RUNS = 6;
	private static final Path REPORT_DIR = Paths.get("build", "reports", "performance");
	private static final Path REPORT_FILE = REPORT_DIR.resolve("hat-kernel-execution.json");

	@Reflect
	public static void vectorAddKernel(@RO KernelContext kc, @RO S32Array left, @RO S32Array right, @WO S32Array out) {
		if (kc.gix < left.length())
			out.array(kc.gix, left.array(kc.gix) + right.array(kc.gix));
	}

	@Reflect
	public static void vectorAdd(@RO ComputeContext cc, @RO S32Array left, @RO S32Array right, @WO S32Array out) {
		cc.dispatchKernel(NDRange.of1D(left.length()), kc -> vectorAddKernel(kc, left, right, out));
	}

	@Test
	void executesKernelAndBenchmarksJavaBackends() throws Exception {
		BackendResult seq = benchmarkVectorAdd(JAVA_SEQ_BACKEND);
		BackendResult mt = benchmarkVectorAdd(JAVA_MT_BACKEND);

		assertTrue(seq.avgNanos > 0L, "Sequential backend produced no timing.");
		assertTrue(mt.avgNanos > 0L, "Multithreaded backend produced no timing.");

		double speedup = (double) seq.avgNanos / (double) mt.avgNanos;
		System.out.printf(Locale.ROOT,
				"HAT vectorAdd (%d elements): seq=%.3f ms, mt=%.3f ms, speedup(seq/mt)=%.3fx%n", VECTOR_LENGTH,
				nanosToMillis(seq.avgNanos), nanosToMillis(mt.avgNanos), speedup);
		Path report = writeReport(seq, mt, speedup);
		assertTrue(Files.exists(report), "Expected performance report to be generated: " + report.toAbsolutePath());
		System.out.println("Wrote performance report: " + report.toAbsolutePath());
	}

	@Reflect
	public static BackendResult benchmarkVectorAdd(String backendClassName) {
		Accelerator accelerator = new Accelerator(MethodHandles.lookup(),
				backend -> backend.getClass().getName().equals(backendClassName));

		S32Array left = S32Array.create(accelerator, VECTOR_LENGTH);
		S32Array right = S32Array.create(accelerator, VECTOR_LENGTH);
		S32Array out = S32Array.create(accelerator, VECTOR_LENGTH);

		for (int i = 0; i < VECTOR_LENGTH; i++) {
			left.array(i, i);
			right.array(i, i + 7);
		}

		for (int i = 0; i < WARMUP_RUNS; i++)
			accelerator.compute(cc -> vectorAdd(cc, left, right, out));

		long totalNanos = 0L;
		for (int i = 0; i < MEASURED_RUNS; i++) {
			long start = System.nanoTime();
			accelerator.compute(cc -> vectorAdd(cc, left, right, out));
			totalNanos += (System.nanoTime() - start);
		}

		for (int i = 0; i < VECTOR_LENGTH; i++) {
			assertEquals((i * 2) + 7, out.array(i),
					"Incorrect output at index " + i + " for backend " + backendClassName);
		}

		return new BackendResult(backendClassName, totalNanos / MEASURED_RUNS);
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
					  "benchmark": "hat-vector-add",
					  "vectorLength": %d,
					  "warmupRuns": %d,
					  "measuredRuns": %d,
					  "sequential": {
					    "backend": "%s",
					    "avgMillis": %.6f
					  },
					  "multithreaded": {
					    "backend": "%s",
					    "avgMillis": %.6f
					  },
					  "speedupSeqOverMt": %.6f
					}
					""",
					Instant.now(), VECTOR_LENGTH, WARMUP_RUNS, MEASURED_RUNS, seq.backendClassName(),
					nanosToMillis(seq.avgNanos()), mt.backendClassName(), nanosToMillis(mt.avgNanos()), speedup);
			Files.writeString(REPORT_FILE, json);
			return REPORT_FILE;
		} catch (IOException e) {
			throw new RuntimeException("Failed to write performance report", e);
		}
	}

	private record BackendResult(String backendClassName, long avgNanos) {
	}
}
