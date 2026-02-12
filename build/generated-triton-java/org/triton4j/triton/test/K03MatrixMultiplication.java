// """
// Matrix Multiplication
// =====================
// In this tutorial, you will write a very short high-performance FP16 matrix multiplication kernel that achieves
// performance on par with cuBLAS or rocBLAS.
//
// You will specifically learn about:
//
// * Block-level matrix multiplications.
//
// * Multi-dimensional pointer arithmetic.
//
// * Program re-ordering for improved L2 cache hit rate.
//
// * Automatic performance tuning.
//
// """
//
package org.triton4j.triton.test;

import static oracle.code.triton.Triton.*;

import java.lang.String;
import jdk.incubator.code.Reflect;
import oracle.code.triton.Constant;
import oracle.code.triton.Ptr;

public class K03MatrixMultiplication {
  /**
   * @param c_ptr  Matrix dimensions
   * @param K  by to get the element one row down (A has M rows).
   * @param stride_ak 
   * @param stride_bn 
   * @param stride_cn  Meta-parameters
   */
  @Reflect
  public void matmulKernel(Ptr a_ptr, Ptr b_ptr, Ptr c_ptr, int M, int N, int K, int stride_am,
      int stride_ak, int stride_bk, int stride_bn, int stride_cm, int stride_cn,
      @Constant int BLOCK_SIZE_M, @Constant int BLOCK_SIZE_N, @Constant int BLOCK_SIZE_K,
      @Constant int GROUP_SIZE_M, @Constant String ACTIVATION) {

    // -----------------------------------------------------------



    // -----------------------------------------------------------
    // Map program ids `pid` to the block of C it should compute.
    // This is done in a grouped ordering to promote L2 data reuse.
    // See above `L2 Cache Optimizations` section for details.
    var pid=oracle.code.triton.Triton.programId(0);

    var num_pid_m=oracle.code.triton.Triton.cdiv(M,BLOCK_SIZE_M);

    var num_pid_n=oracle.code.triton.Triton.cdiv(N,BLOCK_SIZE_N);

    var num_pid_in_group=GROUP_SIZE_M*num_pid_n;

    var group_id=pid/num_pid_in_group;

    var first_pid_m=group_id*GROUP_SIZE_M;

    var group_size_m=Math.min(num_pid_m-first_pid_m,GROUP_SIZE_M);

    var pid_m=first_pid_m+((pid%num_pid_in_group)%group_size_m);

    var pid_n=(pid%num_pid_in_group)/group_size_m;

    // ----------------------------------------------------------
    // Create pointers for the first blocks of A and B.
    // We will advance this pointer as we move in the K direction
    // and accumulate
    // `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    // `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    // See above `Pointer Arithmetic` section for details
    var offs_am=oracle.code.triton.Triton.mod((oracle.code.triton.Triton.add(pid_m*BLOCK_SIZE_M,oracle.code.triton.Triton.arange(0,BLOCK_SIZE_M))),M);

    var offs_bn=oracle.code.triton.Triton.mod((oracle.code.triton.Triton.add(pid_n*BLOCK_SIZE_N,oracle.code.triton.Triton.arange(0,BLOCK_SIZE_N))),N);

    var offs_k=oracle.code.triton.Triton.arange(0,BLOCK_SIZE_K);

    var a_ptrs=oracle.code.triton.Triton.add(a_ptr,(oracle.code.triton.Triton.add(oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_am,1),stride_am),oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_k,0),stride_ak))));

    var b_ptrs=oracle.code.triton.Triton.add(b_ptr,(oracle.code.triton.Triton.add(oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_k,1),stride_bk),oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_bn,0),stride_bn))));

    // -----------------------------------------------------------
    // Iterate to compute a block of the C matrix.
    // We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    // of fp32 values for higher accuracy.
    // `accumulator` will be converted back to fp16 after the loop.
    var accumulator=oracle.code.triton.Triton.zeros(float.class,BLOCK_SIZE_M, BLOCK_SIZE_N);

    for (var k = 0; k < oracle.code.triton.Triton.cdiv(K,BLOCK_SIZE_K); k++) {
    // Load the next block of A and B, generate a mask by checking the K dimension.

    var a=a_ptrs;

    var b=b_ptrs;

    // We accumulate along the K dimension.
    accumulator=oracle.code.triton.Triton.dot(a,b);

    // Advance the ptrs to the next K block.
    a_ptrs=oracle.code.triton.Triton.add(a_ptrs,BLOCK_SIZE_K*stride_ak);

    b_ptrs=oracle.code.triton.Triton.add(b_ptrs,BLOCK_SIZE_K*stride_bk);

    }
    if(java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))) {

    accumulator=accumulator;

    }

    var c=accumulator;

    // -----------------------------------------------------------
    // Write back the block of the output matrix C with masks.
    var offs_cm=oracle.code.triton.Triton.add(pid_m*BLOCK_SIZE_M,oracle.code.triton.Triton.arange(0,BLOCK_SIZE_M));

    var offs_cn=oracle.code.triton.Triton.add(pid_n*BLOCK_SIZE_N,oracle.code.triton.Triton.arange(0,BLOCK_SIZE_N));

    var c_ptrs=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(c_ptr,oracle.code.triton.Triton.mul(stride_cm,oracle.code.triton.Triton.expand(offs_cm,1))),oracle.code.triton.Triton.mul(stride_cn,oracle.code.triton.Triton.expand(offs_cn,0)));

    var c_mask=oracle.code.triton.Triton.and((oracle.code.triton.Triton.compare(oracle.code.triton.Triton.expand(offs_cm,1),M,oracle.code.triton.Triton.CompareKind.LessThan)),(oracle.code.triton.Triton.compare(oracle.code.triton.Triton.expand(offs_cn,0),N,oracle.code.triton.Triton.CompareKind.LessThan)));

    oracle.code.triton.Triton.store(c_ptrs,c,c_mask);
  }
}
