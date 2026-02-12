package org.triton4j.codegen.test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.StringWriter;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Comparator;
import java.util.List;

import javax.tools.Diagnostic;
import javax.tools.DiagnosticCollector;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;

import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.Value;
import org.junit.jupiter.api.Test;
import org.triton4j.codegen.TritonWriter;
import org.triton4j.codegen.TritonWriterContext;

public class GraalPyParityTest {

	private static final String GENERATED_PACKAGE = "org.triton4j.codegen.graalpy";
	private static final String GENERATED_CLASS = "ParityKernels";
	private static final Path GENERATED_SOURCES_DIR = Paths.get("build", "generated-test-sources", "graalpy");
	private static final Path GENERATED_CLASSES_DIR = Paths.get("build", "generated-test-classes", "graalpy");

	private static final String PYTHON_SOURCE = """
			def _jit(fn):
			    return fn

			class _Triton:
			    jit = staticmethod(_jit)

			class _TL:
			    constexpr = int

			triton = _Triton()
			tl = _TL()

			@triton.jit
			def add_one(x: tl.constexpr):
			    return x + 1

			@triton.jit
			def mul_add(x: tl.constexpr, y: tl.constexpr, z: tl.constexpr):
			    return x * y + z
			""";

	@Test
	public void generatedJavaMatchesGraalPyResults() throws Exception {
		resetDirectory(GENERATED_SOURCES_DIR);
		resetDirectory(GENERATED_CLASSES_DIR);

		Path pythonInput = GENERATED_SOURCES_DIR.resolve("parity-input.py");
		Files.createDirectories(pythonInput.getParent());
		Files.writeString(pythonInput, PYTHON_SOURCE);

		TritonWriterContext context = new TritonWriterContext();
		context.packageName = GENERATED_PACKAGE;
		context.comment = "GraalPy parity test input.";

		TritonWriter tritonWriter = new TritonWriter();
		tritonWriter.write(context, pythonInput, GENERATED_SOURCES_DIR, GENERATED_CLASS, context.comment);

		Path generatedSource = GENERATED_SOURCES_DIR.resolve(GENERATED_PACKAGE.replace('.', '/'))
				.resolve(GENERATED_CLASS + ".java");
		assertTrue(Files.exists(generatedSource), "Generated source is missing: " + generatedSource.toAbsolutePath());

		compileGeneratedSources(List.of(generatedSource));

		try (Context polyglot = Context.newBuilder("python").allowAllAccess(true).build();
				URLClassLoader classLoader = new URLClassLoader(new URL[] { GENERATED_CLASSES_DIR.toUri().toURL() },
						GraalPyParityTest.class.getClassLoader())) {
			polyglot.eval("python", PYTHON_SOURCE);

			Class<?> javaClass = Class.forName(GENERATED_PACKAGE + "." + GENERATED_CLASS, true, classLoader);
			Object javaInstance = javaClass.getDeclaredConstructor().newInstance();

			Method addOne = javaClass.getMethod("addOne", int.class);
			assertParity(polyglot, javaInstance, addOne, "add_one", 0);
			assertParity(polyglot, javaInstance, addOne, "add_one", 41);

			Method mulAdd = javaClass.getMethod("mulAdd", int.class, int.class, int.class);
			assertParity(polyglot, javaInstance, mulAdd, "mul_add", 2, 3, 4);
			assertParity(polyglot, javaInstance, mulAdd, "mul_add", 5, 6, 7);
			assertParity(polyglot, javaInstance, mulAdd, "mul_add", 9, 0, 11);
		}
	}

	private static void assertParity(Context polyglot, Object javaTarget, Method javaMethod, String pythonFunction,
			Object... args) throws Exception {
		Value pythonMember = polyglot.getBindings("python").getMember(pythonFunction);
		assertNotNull(pythonMember, "Python function not found: " + pythonFunction);
		assertTrue(pythonMember.canExecute(), "Python member is not executable: " + pythonFunction);

		Value pythonResultValue = pythonMember.execute(args);
		assertTrue(pythonResultValue.fitsInInt(), "Python result is not an int for function: " + pythonFunction);
		int pythonResult = pythonResultValue.asInt();

		Object javaResultValue = javaMethod.invoke(javaTarget, args);
		assertTrue(javaResultValue instanceof Number,
				"Generated Java result is not numeric for method: " + javaMethod.getName());
		int javaResult = ((Number) javaResultValue).intValue();

		assertEquals(pythonResult, javaResult,
				"Parity mismatch for " + pythonFunction + " and " + javaMethod.getName());
	}

	private static void compileGeneratedSources(List<Path> sources) throws Exception {
		JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
		assertNotNull(compiler, "System JavaCompiler is not available");

		Files.createDirectories(GENERATED_CLASSES_DIR);
		DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
		StringWriter compilerOutput = new StringWriter();

		try (StandardJavaFileManager fileManager = compiler.getStandardFileManager(diagnostics, null, null)) {
			Iterable<? extends JavaFileObject> compilationUnits = fileManager.getJavaFileObjectsFromPaths(sources);
			List<String> options = List.of("-classpath", System.getProperty("java.class.path"), "-d",
					GENERATED_CLASSES_DIR.toAbsolutePath().toString(), "--add-modules", "jdk.incubator.code");

			Boolean success = compiler.getTask(compilerOutput, fileManager, diagnostics, options, null, compilationUnits)
					.call();
			if (Boolean.TRUE.equals(success))
				return;

			StringBuilder errorBuilder = new StringBuilder();
			errorBuilder.append("Generated source compilation failed.\n");
			errorBuilder.append(compilerOutput).append('\n');
			for (Diagnostic<? extends JavaFileObject> diagnostic : diagnostics.getDiagnostics()) {
				errorBuilder.append(diagnostic.getKind()).append(": ").append(diagnostic.getMessage(null)).append(" [")
						.append(diagnostic.getSource() == null ? "unknown" : diagnostic.getSource().toUri()).append(':')
						.append(diagnostic.getLineNumber()).append("]\n");
			}
			throw new AssertionError(errorBuilder.toString());
		}
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
