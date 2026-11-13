package org.triton4j.codegen.test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.StringWriter;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
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

import jdk.incubator.code.Op;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaType;
import oracle.code.triton.ConstantType;
import oracle.code.triton.PtrType;
import oracle.code.triton.TritonOps;
import oracle.code.triton.TritonTransformer;

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
		assertCoreKernelSemantics();
		assertCoreKernelsLowerToTriton();
	}

	private void assertCoreKernelsLowerToTriton() throws Exception {
		try (URLClassLoader loader = new URLClassLoader(new URL[] { GENERATED_CLASSES_DIR.toUri().toURL() },
				getClass().getClassLoader())) {
			String vectorIr = lowerGeneratedKernel(loader, "VectorAdd", "addKernel", List.of(
					new PtrType(JavaType.FLOAT), new PtrType(JavaType.FLOAT), new PtrType(JavaType.FLOAT), JavaType.INT,
					new ConstantType(JavaType.INT, 64)));
			assertTrue(countOccurrences(vectorIr, "tt.load") >= 2, "Lowered vector-add must contain both input loads.");
			assertContains(vectorIr, "tt.store", "Lowered vector-add must contain its output store");

			String softmaxIr = lowerGeneratedKernel(loader, "FusedSoftmax", "softmaxKernel", List.of(
					new PtrType(JavaType.FLOAT), new PtrType(JavaType.FLOAT), JavaType.INT, JavaType.INT, JavaType.INT,
					JavaType.INT, new ConstantType(JavaType.INT, 64), new ConstantType(JavaType.INT, 1)));
			assertContains(softmaxIr, "tt.load", "Lowered softmax must contain its masked row load");
			assertContains(softmaxIr, "math.exp", "Lowered softmax must exponentiate its stabilized row");
			assertContains(softmaxIr, "tt.store", "Lowered softmax must contain its output store");

			String matmulIr = lowerGeneratedKernel(loader, "MatrixMultiplication", "matmulKernel", List.of(
					new PtrType(JavaType.FLOAT), new PtrType(JavaType.FLOAT), new PtrType(JavaType.FLOAT),
					JavaType.INT, JavaType.INT, JavaType.INT, JavaType.INT, JavaType.INT, JavaType.INT, JavaType.INT,
					JavaType.INT, JavaType.INT, new ConstantType(JavaType.INT, 32), new ConstantType(JavaType.INT, 64),
					new ConstantType(JavaType.INT, 32), new ConstantType(JavaType.INT, 8),
					new ConstantType(JavaType.BOOLEAN, false)));
			assertTrue(countOccurrences(matmulIr, "tt.load") >= 2, "Lowered matmul must contain both tile loads.");
			assertContains(matmulIr, "tt.dot", "Lowered matmul must contain its block dot product");
			assertContains(matmulIr, "tt.store", "Lowered matmul must contain its masked output store");

			String persistentBaseIr = lowerGeneratedKernel(loader, "PersistentMatmul", "matmulKernel", List.of(
					new PtrType(JavaType.FLOAT), new PtrType(JavaType.FLOAT), new PtrType(JavaType.FLOAT),
					JavaType.INT, JavaType.INT, JavaType.INT, JavaType.INT, JavaType.INT, JavaType.INT, JavaType.INT,
					JavaType.INT, JavaType.INT, new ConstantType(JavaType.INT, 32), new ConstantType(JavaType.INT, 64),
					new ConstantType(JavaType.INT, 32), new ConstantType(JavaType.INT, 8)));
			assertTrue(countOccurrences(persistentBaseIr, "tt.load") >= 2,
					"Persistent tutorial's pointer-based baseline must retain both tile loads.");
			assertContains(persistentBaseIr, "tt.dot", "Persistent tutorial's baseline must retain its dot product");
			assertContains(persistentBaseIr, "tt.store", "Persistent tutorial's baseline must retain its output store");

			String persistentIr = lowerGeneratedKernel(loader, "PersistentMatmul", "matmulKernelPersistent", List.of(
					new PtrType(JavaType.FLOAT), new PtrType(JavaType.FLOAT), new PtrType(JavaType.FLOAT),
					JavaType.INT, JavaType.INT, JavaType.INT, JavaType.INT, JavaType.INT, JavaType.INT, JavaType.INT,
					JavaType.INT, JavaType.INT, new ConstantType(JavaType.INT, 32), new ConstantType(JavaType.INT, 64),
					new ConstantType(JavaType.INT, 32), new ConstantType(JavaType.INT, 8),
					new ConstantType(JavaType.INT, 16)));
			assertContains(persistentIr, "scf.if", "Persistent matmul must lower its scheduling branches");
			assertContains(persistentIr, "tt.dot", "Persistent matmul must retain its dot product after lowering");
			assertContains(persistentIr, "tt.store", "Persistent matmul must retain its output store after lowering");
			assertTrue(!persistentIr.contains("java.if") && !persistentIr.contains("java.cexpression"),
					"Persistent matmul must not retain Java control-flow operations after lowering");
		}
	}

	private String lowerGeneratedKernel(URLClassLoader loader, String className, String methodName,
			List<? extends CodeType> argumentTypes) throws Exception {
		Class<?> generatedClass = Class.forName("org.triton4j.triton.test." + className, true, loader);
		Method method = java.util.Arrays.stream(generatedClass.getDeclaredMethods())
				.filter(candidate -> candidate.getName().equals(methodName)).findFirst().orElseThrow();
		CoreOp.FuncOp reflected = Op.ofMethod(method).orElseThrow();
		TritonOps.ModuleOp module = TritonTransformer.tritonModule(reflected, JavaType.VOID, argumentTypes);
		return module.toText();
	}

	private void assertCoreKernelSemantics() throws Exception {
		String vectorAdd = generatedSource("VectorAdd");
		assertContains(vectorAdd, "Triton.load(Triton.add(x_ptr,offsets),mask)",
				"vector-add must load x through the generated pointer and mask");
		assertContains(vectorAdd, "Triton.load(Triton.add(y_ptr,offsets),mask)",
				"vector-add must load y through the generated pointer and mask");
		assertContains(vectorAdd, "Triton.store(Triton.add(output_ptr,offsets),output,mask)",
				"vector-add must store the computed tensor");

		String softmax = generatedSource("FusedSoftmax");
		assertContains(softmax, "Triton.load(input_ptrs,mask,Float.NEGATIVE_INFINITY)",
				"softmax must mask padded lanes with negative infinity");
		assertContains(softmax, "Triton.max(row,0)", "softmax must reduce the row maximum");
		assertContains(softmax, "Triton.exp(row_minus_max)", "softmax must exponentiate stabilized logits");
		assertContains(softmax, "Triton.sum(numerator,0)", "softmax must compute its normalizer");
		assertContains(softmax, "Triton.store(output_ptrs,softmax_output,mask)",
				"softmax must store only valid lanes");

		String matmul = generatedSource("MatrixMultiplication");
		assertTrue(countOccurrences(matmul, "Triton.load(") >= 2,
				"matmul must retain both masked tile loads");
		assertContains(matmul, "Triton.add(accumulator,Triton.dot(a,b))",
				"matmul must accumulate each dot product instead of replacing the accumulator");
		assertContains(matmul, "Triton.store(c_ptrs,c,c_mask)", "matmul must retain its masked output store");
		assertContains(matmul, "OptionalTritonactivationomitted",
				"basic matmul must state that the optional activation is outside this supported entry point");

		String persistentMatmul = generatedSource("PersistentMatmul");
		assertContains(persistentMatmul, "returnnewObject[]{tile_id,pid_m,pid_n}",
				"persistent matmul must retain the tile helper's tuple result");
		assertContains(persistentMatmul, "if(start_pid<num_tiles%NUM_SMS)",
				"persistent matmul must retain its uneven tile distribution");
		assertContains(persistentMatmul, "ki=(ki==k_tiles-1?0:ki+1)",
				"persistent matmul must advance and wrap its K tile index");
		assertContains(persistentMatmul, "Triton.add(accumulator,Triton.dot(a,b))",
				"persistent matmul must accumulate each dot product");
		assertContains(persistentMatmul, "if(ki==k_tiles-1)",
				"persistent matmul must store only after its final K tile");
		assertContains(persistentMatmul, "Triton.store(c_ptrs,c,c_mask)",
				"persistent matmul must retain its masked output store");
		assertTrue(!persistentMatmul.contains("matmulKernelTmaPersistent")
				&& !persistentMatmul.contains("matmulKernelDescriptorPersistent"),
				"descriptor/TMA kernels are deferred until the Java Triton API supports their operations");
	}

	private String generatedSource(String className) throws Exception {
		Path source = GENERATED_SOURCES_DIR.resolve("org/triton4j/triton/test").resolve(className + ".java");
		assertTrue(Files.exists(source), "Generated source is missing: " + source.toAbsolutePath());
		return Files.readString(source).replace("oracle.code.triton.", "").replaceAll("\\s+", "");
	}

	private void assertContains(String source, String expected, String message) {
		assertTrue(source.contains(expected.replaceAll("\\s+", "")), message + "\nMissing: " + expected);
	}

	private int countOccurrences(String source, String value) {
		int count = 0;
		for (int index = 0; (index = source.indexOf(value, index)) >= 0; index += value.length())
			count++;
		return count;
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
		List<String> failedSources = new ArrayList<>();
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
				failedSources.add(source.getFileName().toString());
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
		failedSources.sort(String::compareTo);
		assertEquals(List.of("FusedAttention.java"), failedSources,
				"The generated-source compilation failure set changed.\n" + allFailures);
		assertTrue(compiledCount >= 8, "Expected at least eight tutorial outputs to compile.");
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
