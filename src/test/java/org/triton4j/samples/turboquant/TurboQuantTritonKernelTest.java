package org.triton4j.samples.turboquant;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Method;
import java.lang.reflect.Parameter;

import org.junit.jupiter.api.Test;

import jdk.incubator.code.Reflect;
import oracle.code.triton.Constant;
import oracle.code.triton.Ptr;

class TurboQuantTritonKernelTest {

	@Test
	void exposesReflectiveKernelEntryPoint() throws Exception {
		Method method = TurboQuantTritonKernels.class.getDeclaredMethod(
				"fusedQkScoresKernel",
				Ptr.class,
				Ptr.class,
				Ptr.class,
				Ptr.class,
				Ptr.class,
				int.class,
				float.class,
				int.class,
				int.class);

		assertNotNull(method.getAnnotation(Reflect.class), "Expected Triton kernel to be annotated with @Reflect.");

		Parameter[] parameters = method.getParameters();
		assertEquals(9, parameters.length, "Expected fused kernel to expose 9 parameters.");
		assertTrue(parameters[7].isAnnotationPresent(Constant.class), "Expected HEAD_DIM to be a @Constant parameter.");
		assertTrue(parameters[8].isAnnotationPresent(Constant.class), "Expected BLOCK_SEQ to be a @Constant parameter.");

		Method attentionMethod = TurboQuantTritonKernels.class.getDeclaredMethod(
				"fusedAttentionKernel",
				Ptr.class,
				Ptr.class,
				Ptr.class,
				Ptr.class,
				Ptr.class,
				Ptr.class,
				int.class,
				float.class,
				int.class,
				int.class);
		assertNotNull(attentionMethod.getAnnotation(Reflect.class), "Expected fused attention kernel to be annotated with @Reflect.");
		Parameter[] attentionParameters = attentionMethod.getParameters();
		assertEquals(10, attentionParameters.length, "Expected fused attention kernel to expose 10 parameters.");
		assertTrue(attentionParameters[8].isAnnotationPresent(Constant.class), "Expected attention HEAD_DIM to be a @Constant parameter.");
		assertTrue(attentionParameters[9].isAnnotationPresent(Constant.class), "Expected attention BLOCK_SEQ to be a @Constant parameter.");
	}
}
