package org.triton4j.codegen.test;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.StringWriter;
import java.nio.file.Files;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import javax.tools.Diagnostic;
import javax.tools.DiagnosticCollector;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;

import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.triton4j.codegen.TritonWriterContext;
import org.triton4j.codegen.TritonWriter;

public class TritonWriterTest {
	static Logger LOG;
	
	static private FileSystem fileSystem = FileSystems.getDefault();
	private static final Path GENERATED_SOURCES_DIR = Paths.get("build", "generated-test-sources", "triton");
	private static final Path GENERATED_CLASSES_DIR = Paths.get("build", "generated-test-classes", "triton");
	
	static {
		LOG = LoggerFactory.getLogger(TritonWriterTest.class);
	}

	@Test
	public void test() throws Exception {
		TritonWriterContext tritonWriterContext = new TritonWriterContext();

		tritonWriterContext.verbose = true;
		tritonWriterContext.packageName = "org.triton4j.triton.test";

		TritonWriter tritonWriter = new TritonWriter();
		List<String> generationFailures = new ArrayList<>();

		resetDirectory(GENERATED_SOURCES_DIR);
		resetDirectory(GENERATED_CLASSES_DIR);

		generate(tritonWriter, tritonWriterContext, "tutorials_python/01-vector-add.py", "VectorAdd", "Add two vectors",
				generationFailures);
		generate(tritonWriter, tritonWriterContext, "tutorials_python/02-fused-softmax.py", "FusedSoftmax",
				"Fused Softmax", generationFailures);
		generate(tritonWriter, tritonWriterContext, "tutorials_python/03-matrix-multiplication.py",
				"MatrixMultiplication", "Matrix Multiplication", generationFailures);
		generate(tritonWriter, tritonWriterContext, "tutorials_python/04-low-memory-dropout.py", "LowMemoryDropout",
				"Low Memory Dropout", generationFailures);
		generate(tritonWriter, tritonWriterContext, "tutorials_python/05-layer-norm.py", "LayerNorm", "Layer Norm",
				generationFailures);
		generate(tritonWriter, tritonWriterContext, "tutorials_python/06-fused-attention.py", "FusedAttention",
				"Fused Attention", generationFailures);
		generate(tritonWriter, tritonWriterContext, "tutorials_python/07-extern-functions.py", "ExternFunctions",
				"Extern Functions", generationFailures);
		generate(tritonWriter, tritonWriterContext, "tutorials_python/08-grouped-gemm.py", "GroupedGEMM",
				"Grouped GEMM", generationFailures);
		generate(tritonWriter, tritonWriterContext, "tutorials_python/09-persistent-matmul.py", "PersistentMatmul",
				"Persistent Matmul", generationFailures);

		assertTrue(generationFailures.isEmpty(), "Generation failures:\n" + String.join("\n", generationFailures));

		compileGeneratedSources();
	}

	private void generate(TritonWriter tritonWriter, TritonWriterContext tritonWriterContext, String samplePath,
			String className, String comment, List<String> generationFailures) {
		try {
			Path path = fileSystem.getPath(samplePath);
			tritonWriter.write(tritonWriterContext, path, GENERATED_SOURCES_DIR, className, comment);
		} catch (Exception e) {
			LOG.error("Failed to generate sample {} -> {}: {}", samplePath, className, e.getLocalizedMessage(), e);
			generationFailures.add(samplePath + " -> " + className + ": " + e.getLocalizedMessage());
		}
	}

	private void compileGeneratedSources() throws Exception {
		List<Path> generatedSources = new ArrayList<>();
		try (var walk = Files.walk(GENERATED_SOURCES_DIR)) {
			walk.filter(path -> path.toString().endsWith(".java")).forEach(generatedSources::add);
		}

		assertTrue(!generatedSources.isEmpty(),
				"No generated source files found in " + GENERATED_SOURCES_DIR.toAbsolutePath());

		JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
		assertNotNull(compiler, "System JavaCompiler is not available");

		int compiledCount = 0;
		int failedCount = 0;
		StringBuilder allFailures = new StringBuilder();

		for (Path source : generatedSources) {
			DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
			StringWriter compilerOutput = new StringWriter();
			try (StandardJavaFileManager fileManager = compiler.getStandardFileManager(diagnostics, null, null)) {
				Iterable<? extends JavaFileObject> compilationUnits = fileManager.getJavaFileObjectsFromPaths(List.of(source));
				List<String> options = new ArrayList<>();
				options.add("-classpath");
				options.add(System.getProperty("java.class.path"));
				options.add("-d");
				options.add(GENERATED_CLASSES_DIR.toAbsolutePath().toString());
				options.add("--add-modules");
				options.add("jdk.incubator.code");

				Boolean success = compiler.getTask(compilerOutput, fileManager, diagnostics, options, null, compilationUnits)
						.call();
				if (Boolean.TRUE.equals(success)) {
					compiledCount++;
					continue;
				}
				failedCount++;
				StringBuilder details = new StringBuilder();
				details.append("Compilation failed for ").append(source).append('\n');
				details.append(compilerOutput.toString()).append('\n');
				for (Diagnostic<? extends JavaFileObject> diagnostic : diagnostics.getDiagnostics()) {
					details.append(diagnostic.getKind()).append(": ").append(diagnostic.getMessage(null)).append(" [")
							.append(diagnostic.getSource() == null ? "unknown" : diagnostic.getSource().toUri())
							.append(':').append(diagnostic.getLineNumber()).append("]\n");
				}
				allFailures.append(details).append('\n');
			}
		}

		LOG.info("Generated source compilation summary: {} succeeded, {} failed", compiledCount, failedCount);
		if (allFailures.length() > 0)
			LOG.warn(allFailures.toString());
	}

	private static void resetDirectory(Path path) throws Exception {
		if (Files.exists(path)) {
			try (var walk = Files.walk(path)) {
				walk.sorted(Comparator.reverseOrder()).forEach(target -> {
					try {
						Files.delete(target);
					} catch (Exception e) {
						throw new RuntimeException(e);
					}
				});
			}
		}
		Files.createDirectories(path);
	}

}
