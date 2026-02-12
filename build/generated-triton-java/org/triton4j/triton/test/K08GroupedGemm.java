// """
// Group GEMM
// ============================
// This group gemm kernel launches a fixed number of CTA to compute a group
// of gemms. The scheduling is static and we do it on device.
// """
//
package org.triton4j.triton.test;

import static oracle.code.triton.Triton.*;

import jdk.incubator.code.Reflect;
import oracle.code.triton.Constant;
import oracle.code.triton.Tensor;

public class K08GroupedGemm {
  /**
   * @param group_c_ptrs  dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
   * @param group_gemm_sizes  dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
   * @param g_lds  number of gemms
   * @param group_size  number of virtual SM
   */
  @Reflect
  public void groupedMatmulKernel(Tensor group_a_ptrs, Tensor group_b_ptrs, Tensor group_c_ptrs,
      int group_gemm_sizes, int g_lds, int group_size, @Constant int NUM_SM,
      @Constant int BLOCK_SIZE_M, @Constant int BLOCK_SIZE_N, @Constant int BLOCK_SIZE_K) {

    // get the gemm size of the current problem

    var tile_idx=oracle.code.triton.Triton.programId(0);

    var last_problem_end=0;

    for (var g = 0; g < group_size; g++) {
    // get the gemm size of the current problem

    var gm=oracle.code.triton.Triton.add(group_gemm_sizes,oracle.code.triton.Triton.mul(g,3));

    var gn=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(group_gemm_sizes,oracle.code.triton.Triton.mul(g,3)),1);

    var gk=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(group_gemm_sizes,oracle.code.triton.Triton.mul(g,3)),2);

    var num_m_tiles=oracle.code.triton.Triton.cdiv(gm,BLOCK_SIZE_M);

    var num_n_tiles=oracle.code.triton.Triton.cdiv(gn,BLOCK_SIZE_N);

    var num_tiles=num_m_tiles*num_n_tiles;

    while((tile_idx>=last_problem_end&&tile_idx<last_problem_end+num_tiles)){

    // pick up a tile from the current gemm problem

    var k=gk;

    var lda=oracle.code.triton.Triton.add(g_lds,oracle.code.triton.Triton.mul(g,3));

    var ldb=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(g_lds,oracle.code.triton.Triton.mul(g,3)),1);

    var ldc=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(g_lds,oracle.code.triton.Triton.mul(g,3)),2);

    var a_ptr=oracle.code.triton.Triton.zeros(float.class,1);

    var b_ptr=oracle.code.triton.Triton.zeros(float.class,1);

    var c_ptr=oracle.code.triton.Triton.zeros(float.class,1);

    // figure out tile coordinates
    var tile_idx_in_gemm=tile_idx-last_problem_end;

    var tile_m_idx=tile_idx_in_gemm/num_n_tiles;

    var tile_n_idx=tile_idx_in_gemm%num_n_tiles;

    // do regular gemm here
    var offs_am=oracle.code.triton.Triton.add(tile_m_idx*BLOCK_SIZE_M,oracle.code.triton.Triton.arange(0,BLOCK_SIZE_M));

    var offs_bn=oracle.code.triton.Triton.add(tile_n_idx*BLOCK_SIZE_N,oracle.code.triton.Triton.arange(0,BLOCK_SIZE_N));

    var offs_k=oracle.code.triton.Triton.arange(0,BLOCK_SIZE_K);

    var a_ptrs=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(a_ptr,oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_am,1),lda)),oracle.code.triton.Triton.expand(offs_k,0));

    var b_ptrs=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(b_ptr,oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_k,1),ldb)),oracle.code.triton.Triton.expand(offs_bn,0));

    var accumulator=oracle.code.triton.Triton.zeros(float.class,BLOCK_SIZE_M, BLOCK_SIZE_N);

    for (var kk = 0; kk < oracle.code.triton.Triton.cdiv(k,BLOCK_SIZE_K); kk++) {
    // hint to Triton compiler to do proper loop pipelining

    java.util.Objects.requireNonNull(a_ptrs);

    java.util.Objects.requireNonNull(b_ptrs);

    // assume full tile for now
    var a=a_ptrs;

    var b=b_ptrs;

    accumulator=oracle.code.triton.Triton.add(accumulator,oracle.code.triton.Triton.dot(a,b));

    a_ptrs=oracle.code.triton.Triton.add(a_ptrs,BLOCK_SIZE_K);

    b_ptrs=oracle.code.triton.Triton.add(b_ptrs,oracle.code.triton.Triton.mul(BLOCK_SIZE_K,ldb));

    }
    var c=accumulator;

    var offs_cm=oracle.code.triton.Triton.add(tile_m_idx*BLOCK_SIZE_M,oracle.code.triton.Triton.arange(0,BLOCK_SIZE_M));

    var offs_cn=oracle.code.triton.Triton.add(tile_n_idx*BLOCK_SIZE_N,oracle.code.triton.Triton.arange(0,BLOCK_SIZE_N));

    var c_ptrs=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(c_ptr,oracle.code.triton.Triton.mul(ldc,oracle.code.triton.Triton.expand(offs_cm,1))),oracle.code.triton.Triton.expand(offs_cn,0));

    // assumes full tile for now
    java.util.Objects.requireNonNull(c_ptrs);

    // go to the next tile by advancing NUM_SM
    tile_idx=tile_idx+NUM_SM;

    }
    last_problem_end=last_problem_end+num_tiles;

    }
  }
}
