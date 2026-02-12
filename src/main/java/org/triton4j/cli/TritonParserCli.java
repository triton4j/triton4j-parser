package org.triton4j.cli;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
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
		subcommands = { TritonParserCli.GenerateCommand.class })
public final class TritonParserCli implements Runnable {

	public static void main(String[] args) {
		int exitCode = new CommandLine(new TritonParserCli()).execute(args);
		System.exit(exitCode);
	}

	@Override
	public void run() {
		CommandLine.usage(this, System.out);
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
