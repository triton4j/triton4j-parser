package org.triton4j.samples.turboquant;

import static oracle.code.triton.Triton.CompareKind.LessThan;
import static oracle.code.triton.Triton.add;
import static oracle.code.triton.Triton.and;
import static oracle.code.triton.Triton.arange;
import static oracle.code.triton.Triton.compare;
import static oracle.code.triton.Triton.expand;
import static oracle.code.triton.Triton.load;
import static oracle.code.triton.Triton.mul;
import static oracle.code.triton.Triton.programId;
import static oracle.code.triton.Triton.store;
import static oracle.code.triton.Triton.sum;
import static oracle.code.triton.Triton.div;
import static oracle.code.triton.Triton.exp;
import static oracle.code.triton.Triton.max;

import jdk.incubator.code.Reflect;
import oracle.code.triton.Constant;
import oracle.code.triton.Ptr;

public final class TurboQuantTritonKernels {

	private TurboQuantTritonKernels() {
	}

	@Reflect
	public static void fusedQkScoresKernel(
			Ptr qPtr,
			Ptr keyIndexPtr,
			Ptr keyNormPtr,
			Ptr centroidPtr,
			Ptr outPtr,
			int seqLen,
			float scale,
			@Constant int HEAD_DIM,
			@Constant int BLOCK_SEQ) {
		var pid = programId(0);
		var sOffsets = add(pid * BLOCK_SEQ, arange(0, BLOCK_SEQ));
		var sMask = compare(sOffsets, seqLen, LessThan);

		var dOffsets = arange(0, HEAD_DIM);
		var dMask = compare(dOffsets, HEAD_DIM, LessThan);
		var query = load(add(qPtr, dOffsets), dMask, 0.0f);

		var sExpanded = expand(sOffsets, 1);
		var dExpanded = expand(dOffsets, 0);
		var keyOffsets = add(mul(sExpanded, HEAD_DIM), dExpanded);
		var keyMask = and(compare(sExpanded, seqLen, LessThan), compare(dExpanded, HEAD_DIM, LessThan));

		var keyIndices = load(add(keyIndexPtr, keyOffsets), keyMask, 0.0f);
		var centroidPtrs = add(centroidPtr, keyIndices);
		var gatheredCentroids = load(centroidPtrs, keyMask, 0.0f);

		var queryExpanded = expand(query, 0);
		var weighted = mul(gatheredCentroids, queryExpanded);
		var dotProducts = sum(weighted, 1);

		var norms = load(add(keyNormPtr, sOffsets), sMask, 0.0f);
		var scores = mul(mul(norms, dotProducts), scale);
		store(add(outPtr, sOffsets), scores, sMask);
	}

	@Reflect
	public static void fusedAttentionKernel(
			Ptr qPtr,
			Ptr keyIndexPtr,
			Ptr keyNormPtr,
			Ptr centroidPtr,
			Ptr valuePtr,
			Ptr outPtr,
			int seqLen,
			float scale,
			@Constant int HEAD_DIM,
			@Constant int BLOCK_SEQ) {
		var pid = programId(0);
		var sOffsets = add(pid * BLOCK_SEQ, arange(0, BLOCK_SEQ));
		var sMask = compare(sOffsets, seqLen, LessThan);

		var dOffsets = arange(0, HEAD_DIM);
		var dMask = compare(dOffsets, HEAD_DIM, LessThan);
		var query = load(add(qPtr, dOffsets), dMask, 0.0f);

		var sExpanded = expand(sOffsets, 1);
		var dExpanded = expand(dOffsets, 0);
		var keyOffsets = add(mul(sExpanded, HEAD_DIM), dExpanded);
		var keyMask = and(compare(sExpanded, seqLen, LessThan), compare(dExpanded, HEAD_DIM, LessThan));

		var keyIndices = load(add(keyIndexPtr, keyOffsets), keyMask, 0.0f);
		var centroidPtrs = add(centroidPtr, keyIndices);
		var gatheredCentroids = load(centroidPtrs, keyMask, 0.0f);
		var queryExpanded = expand(query, 0);
		var weightedKeys = mul(gatheredCentroids, queryExpanded);
		var dotProducts = sum(weightedKeys, 1);
		var norms = load(add(keyNormPtr, sOffsets), sMask, 0.0f);
		var scores = mul(mul(norms, dotProducts), scale);

		var maxScore = max(scores, 0);
		var stabilized = add(scores, mul(maxScore, -1.0f));
		var logits = exp(stabilized);
		var normalizer = sum(logits, 0);
		var weights = div(logits, normalizer);

		var valueOffsets = add(mul(sExpanded, HEAD_DIM), dExpanded);
		var valueMask = and(compare(sExpanded, seqLen, LessThan), compare(dExpanded, HEAD_DIM, LessThan));
		var values = load(add(valuePtr, valueOffsets), valueMask, 0.0f);
		var weightsExpanded = expand(weights, 1);
		var weightedValues = mul(values, weightsExpanded);
		var output = sum(weightedValues, 0);
		store(add(outPtr, dOffsets), output, dMask);
	}
}