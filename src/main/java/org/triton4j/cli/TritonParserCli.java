/*
 * Copyright 2025 dScope, LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

package org.triton4j.cli;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.Callable;

import org.triton4j.codegen.TritonWriter;
import org.triton4j.codegen.TritonWriterContext;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Model.CommandSpec;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import picocli.CommandLine.ParameterException;
import picocli.CommandLine.Spec;

@Command(name = "triton-parser", mixinStandardHelpOptions = true, version = "0.1.0",
		description = "Parse Triton-style Python kernels and generate Java source.",
		subcommands = { TritonParserCli.GenerateCommand.class, TritonParserCli.BenchmarkCommand.class })
public final class TritonParserCli implements Runnable {

	public static void main(String[] args) {
		int exitCode = new CommandLine(new TritonParserCli()).execute(args);
		System.exit(exitCode);
	}

	@Override
	public void run() {
		CommandLine.usage(this, System.out);
	}

	@Command(name = "benchmark", mixinStandardHelpOptions = true,
			description = "Run paired benchmark commands (GPU-mode and CPU/non-GPU mode) and compare timings.")
	static final class BenchmarkCommand implements Callable<Integer> {
		@Option(names = { "--gpu-cmd" }, required = true, paramLabel = "CMD",
				description = "Shell command for the GPU-enabled benchmark case.")
		private String gpuCommand;

		@Option(names = { "--cpu-cmd" }, required = true, paramLabel = "CMD",
				description = "Shell command for the CPU/non-GPU benchmark case.")
		private String cpuCommand;

		@Option(names = { "--warmup" }, defaultValue = "2", paramLabel = "N",
				description = "Warmup runs per case. Default: ${DEFAULT-VALUE}.")
		private int warmupRuns;

		@Option(names = { "--iterations" }, defaultValue = "8", paramLabel = "N",
				description = "Measured runs per case. Default: ${DEFAULT-VALUE}.")
		private int measuredRuns;

		@Option(names = { "--workdir" }, defaultValue = ".", paramLabel = "DIR",
				description = "Working directory for benchmark commands. Default: current directory.")
		private Path workDir;

		@Option(names = { "--env" }, paramLabel = "KEY=VALUE",
				description = "Additional environment variable(s) passed to both commands.")
		private List<String> envEntries = new ArrayList<>();

		@Option(names = { "--verbose" }, description = "Print command output for each run.")
		private boolean verbose;

		@Option(names = { "--report-file" }, paramLabel = "FILE",
				description = "Optional JSON report output file for benchmark results.")
		private Path reportFile;

		@Spec
		private CommandSpec spec;

		@Override
		public Integer call() throws Exception {
			if (warmupRuns < 0)
				throw new ParameterException(spec.commandLine(), "--warmup must be >= 0");
			if (measuredRuns <= 0)
				throw new ParameterException(spec.commandLine(), "--iterations must be > 0");

			Path absoluteWorkDir = workDir.toAbsolutePath().normalize();
			if (!Files.isDirectory(absoluteWorkDir))
				throw new ParameterException(spec.commandLine(), "--workdir is not a directory: " + absoluteWorkDir);

			Map<String, String> extraEnv = parseEnvEntries(envEntries);

			spec.commandLine().getOut().printf("Benchmark workdir: %s%n", absoluteWorkDir);
			spec.commandLine().getOut().printf("Warmup=%d, iterations=%d%n", warmupRuns, measuredRuns);

			BenchmarkResult gpuResult = runCase("GPU", gpuCommand, absoluteWorkDir, extraEnv, warmupRuns, measuredRuns);
			BenchmarkResult cpuResult = runCase("CPU", cpuCommand, absoluteWorkDir, extraEnv, warmupRuns, measuredRuns);

			double speedup = printSummary(gpuResult, cpuResult);
			writeReportIfRequested(reportFile, absoluteWorkDir, extraEnv, gpuResult, cpuResult, speedup);
			return 0;
		}

		private Map<String, String> parseEnvEntries(List<String> entries) {
			Map<String, String> parsed = new HashMap<>();
			for (String entry : entries) {
				if (entry == null || entry.isBlank())
					continue;
				int idx = entry.indexOf('=');
				if (idx <= 0 || idx == entry.length() - 1) {
					throw new ParameterException(spec.commandLine(), "Invalid --env entry (expected KEY=VALUE): " + entry);
				}
				String key = entry.substring(0, idx).trim();
				String value = entry.substring(idx + 1);
				if (key.isEmpty())
					throw new ParameterException(spec.commandLine(), "Invalid --env key in entry: " + entry);
				parsed.put(key, value);
			}
			return parsed;
		}

		private BenchmarkResult runCase(String name, String command, Path absoluteWorkDir, Map<String, String> extraEnv,
				int warmup, int iterations) throws Exception {
			spec.commandLine().getOut().printf("%n[%s] command: %s%n", name, command);

			for (int i = 0; i < warmup; i++) {
				CommandRun run = execute(command, absoluteWorkDir, extraEnv);
				if (run.exitCode != 0)
					throw new ParameterException(spec.commandLine(),
							String.format(Locale.ROOT, "[%s] warmup %d failed with exit code %d", name, i + 1, run.exitCode));
				if (verbose) {
					spec.commandLine().getOut().printf("[%s] warmup %d output:%n%s%n", name, i + 1, run.output);
				}
			}

			List<Long> measurements = new ArrayList<>(iterations);
			for (int i = 0; i < iterations; i++) {
				CommandRun run = execute(command, absoluteWorkDir, extraEnv);
				if (run.exitCode != 0)
					throw new ParameterException(spec.commandLine(),
							String.format(Locale.ROOT, "[%s] iteration %d failed with exit code %d", name, i + 1, run.exitCode));
				measurements.add(run.durationNanos);
				spec.commandLine().getOut().printf("[%s] iteration %d: %.3f ms%n", name, i + 1, nanosToMillis(run.durationNanos));
				if (verbose) {
					spec.commandLine().getOut().printf("[%s] iteration %d output:%n%s%n", name, i + 1, run.output);
				}
			}

			return BenchmarkResult.from(name, measurements);
		}

		private CommandRun execute(String command, Path absoluteWorkDir, Map<String, String> extraEnv)
				throws IOException, InterruptedException {
			ProcessBuilder processBuilder = new ProcessBuilder(resolveShellCommand(command));
			processBuilder.directory(absoluteWorkDir.toFile());
			processBuilder.redirectErrorStream(true);
			processBuilder.environment().putAll(extraEnv);

			long start = System.nanoTime();
			Process process = processBuilder.start();
			byte[] outputBytes = process.getInputStream().readAllBytes();
			int exitCode = process.waitFor();
			long duration = System.nanoTime() - start;

			String output = new String(outputBytes, StandardCharsets.UTF_8);
			return new CommandRun(exitCode, duration, output);
		}

		private List<String> resolveShellCommand(String rawCommand) {
			String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
			if (os.contains("win"))
				return List.of("cmd", "/c", rawCommand);
			return List.of("/bin/zsh", "-lc", rawCommand);
		}

		private double printSummary(BenchmarkResult gpuResult, BenchmarkResult cpuResult) {
			spec.commandLine().getOut().printf("%nSummary:%n");
			spec.commandLine().getOut().printf("%s avg=%.3f ms min=%.3f ms max=%.3f ms%n", gpuResult.name,
					nanosToMillis(gpuResult.avgNanos), nanosToMillis(gpuResult.minNanos), nanosToMillis(gpuResult.maxNanos));
			spec.commandLine().getOut().printf("%s avg=%.3f ms min=%.3f ms max=%.3f ms%n", cpuResult.name,
					nanosToMillis(cpuResult.avgNanos), nanosToMillis(cpuResult.minNanos), nanosToMillis(cpuResult.maxNanos));

			double speedup = ((double) cpuResult.avgNanos) / ((double) gpuResult.avgNanos);
			spec.commandLine().getOut().printf(Locale.ROOT, "Speedup (CPU avg / GPU avg): %.3fx%n", speedup);
			return speedup;
		}

		private void writeReportIfRequested(Path maybeReportFile, Path absoluteWorkDir, Map<String, String> extraEnv,
				BenchmarkResult gpuResult, BenchmarkResult cpuResult, double speedup) throws IOException {
			if (maybeReportFile == null)
				return;

			Path absoluteReportFile = maybeReportFile.toAbsolutePath().normalize();
			Path reportParent = absoluteReportFile.getParent();
			if (reportParent != null)
				Files.createDirectories(reportParent);

			String report = buildReportJson(absoluteWorkDir, extraEnv, gpuResult, cpuResult, speedup);
			Files.writeString(absoluteReportFile, report, StandardCharsets.UTF_8);
			spec.commandLine().getOut().printf("Wrote benchmark report: %s%n", absoluteReportFile);
		}

		private String buildReportJson(Path absoluteWorkDir, Map<String, String> extraEnv, BenchmarkResult gpuResult,
				BenchmarkResult cpuResult, double speedup) {
			StringBuilder json = new StringBuilder(2048);
			json.append("{\n");
			json.append("  \"timestamp\": \"").append(escapeJson(Instant.now().toString())).append("\",\n");
			json.append("  \"workdir\": \"").append(escapeJson(absoluteWorkDir.toString())).append("\",\n");
			json.append("  \"warmupRuns\": ").append(warmupRuns).append(",\n");
			json.append("  \"iterations\": ").append(measuredRuns).append(",\n");
			json.append("  \"commands\": {\n");
			json.append("    \"gpu\": \"").append(escapeJson(gpuCommand)).append("\",\n");
			json.append("    \"cpu\": \"").append(escapeJson(cpuCommand)).append("\"\n");
			json.append("  },\n");
			json.append("  \"env\": ").append(formatEnvJson(extraEnv)).append(",\n");
			json.append("  \"results\": {\n");
			json.append("    \"gpu\": ").append(formatResultJson(gpuResult)).append(",\n");
			json.append("    \"cpu\": ").append(formatResultJson(cpuResult)).append("\n");
			json.append("  },\n");
			json.append(String.format(Locale.ROOT, "  \"speedupCpuOverGpu\": %.6f%n", speedup));
			json.append("}\n");
			return json.toString();
		}

		private String formatEnvJson(Map<String, String> env) {
			if (env.isEmpty())
				return "{}";

			TreeMap<String, String> sorted = new TreeMap<>(env);
			StringBuilder json = new StringBuilder();
			json.append("{\n");
			int index = 0;
			for (Map.Entry<String, String> entry : sorted.entrySet()) {
				json.append("    \"").append(escapeJson(entry.getKey())).append("\": \"")
						.append(escapeJson(entry.getValue())).append("\"");
				if (index < sorted.size() - 1)
					json.append(',');
				json.append('\n');
				index++;
			}
			json.append("  }");
			return json.toString();
		}

		private String formatResultJson(BenchmarkResult result) {
			StringBuilder json = new StringBuilder();
			json.append("{\n");
			json.append("      \"name\": \"").append(escapeJson(result.name)).append("\",\n");
			json.append(String.format(Locale.ROOT, "      \"avgMillis\": %.6f,%n", nanosToMillis(result.avgNanos)));
			json.append(String.format(Locale.ROOT, "      \"minMillis\": %.6f,%n", nanosToMillis(result.minNanos)));
			json.append(String.format(Locale.ROOT, "      \"maxMillis\": %.6f,%n", nanosToMillis(result.maxNanos)));
			json.append("      \"iterationsMillis\": [");
			for (int i = 0; i < result.measurementsNanos.size(); i++) {
				if (i > 0)
					json.append(", ");
				json.append(String.format(Locale.ROOT, "%.6f", nanosToMillis(result.measurementsNanos.get(i))));
			}
			json.append("]\n");
			json.append("    }");
			return json.toString();
		}

		private String escapeJson(String value) {
			StringBuilder escaped = new StringBuilder(value.length() + 16);
			for (int i = 0; i < value.length(); i++) {
				char c = value.charAt(i);
				switch (c) {
				case '\\':
					escaped.append("\\\\");
					break;
				case '"':
					escaped.append("\\\"");
					break;
				case '\n':
					escaped.append("\\n");
					break;
				case '\r':
					escaped.append("\\r");
					break;
				case '\t':
					escaped.append("\\t");
					break;
				default:
					if (c < 0x20) {
						escaped.append(String.format(Locale.ROOT, "\\u%04x", (int) c));
					} else {
						escaped.append(c);
					}
				}
			}
			return escaped.toString();
		}

		private static double nanosToMillis(long nanos) {
			return nanos / 1_000_000.0d;
		}

		private static final class CommandRun {
			final int exitCode;
			final long durationNanos;
			final String output;

			CommandRun(int exitCode, long durationNanos, String output) {
				this.exitCode = exitCode;
				this.durationNanos = durationNanos;
				this.output = output;
			}
		}

		private static final class BenchmarkResult {
			final String name;
			final long minNanos;
			final long maxNanos;
			final long avgNanos;
			final List<Long> measurementsNanos;

			private BenchmarkResult(String name, long minNanos, long maxNanos, long avgNanos, List<Long> measurementsNanos) {
				this.name = name;
				this.minNanos = minNanos;
				this.maxNanos = maxNanos;
				this.avgNanos = avgNanos;
				this.measurementsNanos = measurementsNanos;
			}

			static BenchmarkResult from(String name, List<Long> values) {
				long min = Long.MAX_VALUE;
				long max = Long.MIN_VALUE;
				long sum = 0L;
				for (Long value : values) {
					long v = value.longValue();
					if (v < min)
						min = v;
					if (v > max)
						max = v;
					sum += v;
				}
				long avg = sum / values.size();
				return new BenchmarkResult(name, min, max, avg, List.copyOf(values));
			}
		}
	}

	@Command(name = "generate", mixinStandardHelpOptions = true,
			description = "Generate Java code from one or more Python files/directories.")
	static final class GenerateCommand implements Callable<Integer> {
		@Parameters(arity = "1..*", paramLabel = "INPUT",
				description = "Python file(s) or directory/directories containing .py files.")
		private List<Path> inputs;

		@Option(names = { "-o", "--output" }, required = true, paramLabel = "DIR",
				description = "Output directory for generated Java sources.")
		private Path outputDir;

		@Option(names = { "-p", "--package" }, required = true, paramLabel = "NAME",
				description = "Java package name for generated sources.")
		private String packageName;

		@Option(names = { "-c", "--class" }, paramLabel = "NAME",
				description = "Class name (only valid when a single input file is resolved).")
		private String className;

		@Option(names = { "--comment" }, paramLabel = "TEXT",
				description = "Java file header comment. Defaults to 'Generated from <filename>'.")
		private String comment;

		@Option(names = { "-l", "--language" }, defaultValue = "tl", paramLabel = "NAME",
				description = "Triton language alias used in Python source. Default: ${DEFAULT-VALUE}.")
		private String language;

		@Option(names = { "-v", "--verbose" }, description = "Print generated source to stdout.")
		private boolean verbose;

		@Option(names = { "--continue-on-error" },
				description = "Continue generating remaining files when one input fails.")
		private boolean continueOnError;

		@Spec
		private CommandSpec spec;

		@Override
		public Integer call() throws Exception {
			List<Path> sources = resolveSources(inputs);
			if (sources.isEmpty())
				throw new ParameterException(spec.commandLine(), "No .py source files were found.");
			if (className != null && sources.size() != 1) {
				throw new ParameterException(spec.commandLine(),
						"--class can be used only when exactly one source file is resolved.");
			}

			Files.createDirectories(outputDir);
			TritonWriter writer = new TritonWriter();
			int failures = 0;

			for (Path source : sources) {
				String targetClass = className != null ? className : deriveClassName(source);
				String fileComment = comment != null ? comment : "Generated from " + source.getFileName();
				TritonWriterContext context = new TritonWriterContext();
				context.verbose = verbose;
				context.packageName = packageName;
				context.language = language;
				context.comment = fileComment;

				try {
					writer.write(context, source, outputDir, targetClass, fileComment);
					spec.commandLine().getOut().printf("Generated %s from %s%n", targetClass, source);
				} catch (Exception e) {
					failures++;
					spec.commandLine().getErr().printf("Failed to generate %s from %s: %s%n", targetClass, source,
							e.getMessage());
					if (!continueOnError)
						return 2;
				}
			}

			return failures == 0 ? 0 : 2;
		}

		private List<Path> resolveSources(List<Path> rawInputs) throws IOException {
			List<Path> sources = new ArrayList<>();
			for (Path input : rawInputs) {
				if (!Files.exists(input))
					throw new ParameterException(spec.commandLine(), "Input path does not exist: " + input);

				if (Files.isRegularFile(input)) {
					if (!isPythonFile(input))
						throw new ParameterException(spec.commandLine(),
								"Input file must have .py extension: " + input);
					sources.add(input);
					continue;
				}

				if (Files.isDirectory(input)) {
					try (var walk = Files.walk(input)) {
						walk.filter(Files::isRegularFile).filter(this::isPythonFile).forEach(sources::add);
					}
					continue;
				}

				throw new ParameterException(spec.commandLine(), "Unsupported input type: " + input);
			}

			sources.sort(Comparator.naturalOrder());
			return sources;
		}

		private boolean isPythonFile(Path path) {
			String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
			return name.endsWith(".py");
		}

		private String deriveClassName(Path path) {
			String fileName = path.getFileName().toString();
			int dot = fileName.lastIndexOf('.');
			String base = dot > 0 ? fileName.substring(0, dot) : fileName;
			String[] parts = base.split("[^A-Za-z0-9]+");
			StringBuilder classBuilder = new StringBuilder();
			for (String part : parts) {
				if (part.isBlank())
					continue;
				String lower = part.toLowerCase(Locale.ROOT);
				classBuilder.append(Character.toUpperCase(lower.charAt(0)));
				if (lower.length() > 1)
					classBuilder.append(lower.substring(1));
			}
			if (classBuilder.length() == 0)
				classBuilder.append("GeneratedKernel");
			if (Character.isDigit(classBuilder.charAt(0)))
				classBuilder.insert(0, "K");
			return classBuilder.toString();
		}
	}
}
