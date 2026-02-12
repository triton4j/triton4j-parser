// """
// Persistent Matmul
// =====================
// This script demonstrates persistent kernel implementations of matrix multiplication using Triton.
// Various matmul methods are included, such as naive, persistent, and TMA (Tensor Memory Accelerator) based approaches.
// The kernels support both FP16 and FP8 data types but the FP8 implementation is only available on CUDA devices with compute capability >= 9.0.
//
// Triton and cuBLAS implementations are benchmarked under different configurations and evaluated using the proton profiler.
// Users can pass command-line arguments to specify matrix dimensions and iteration steps flexibly.
//
// .. code-block:: bash
//
//     # FP8
//     python 09-persistent-matmul.py --prec fp8 --K_range 128 1024 --K_step 128
//
//     # FP16
//     python 09-persistent-matmul.py --prec fp16 --K_range 128 1024 --K_step 128
//
// Note that currently this tutorial will fail on devices with a small shared memory size, such as RTX-4090.
// """
//
package org.triton4j.triton.test;

import static oracle.code.triton.Triton.*;

import java.lang.Object;
import jdk.incubator.code.Reflect;
import oracle.code.triton.Constant;
import oracle.code.triton.Ptr;

public class PersistentMatmul {
  /**
   * @param c_ptr 
   * @param K 
   * @param stride_ak 
   * @param stride_bn 
   * @param stride_cn 
   */
  @Reflect
  public void matmulKernel(Ptr a_ptr, Ptr b_ptr, Ptr c_ptr, int M, int N, int K, int stride_am,
      int stride_ak, int stride_bk, int stride_bn, int stride_cm, int stride_cn,
      @Constant int BLOCK_SIZE_M, @Constant int BLOCK_SIZE_N, @Constant int BLOCK_SIZE_K,
      @Constant int GROUP_SIZE_M) {


    var pid=oracle.code.triton.Triton.programId(0);

    var num_pid_m=oracle.code.triton.Triton.cdiv(M,BLOCK_SIZE_M);

    var num_pid_n=oracle.code.triton.Triton.cdiv(N,BLOCK_SIZE_N);

    var num_pid_in_group=GROUP_SIZE_M*num_pid_n;

    var group_id=pid/num_pid_in_group;

    var first_pid_m=group_id*GROUP_SIZE_M;

    var group_size_m=Math.min(num_pid_m-first_pid_m,GROUP_SIZE_M);

    var pid_m=first_pid_m+(pid%group_size_m);

    var pid_n=(pid%num_pid_in_group)/group_size_m;

    var start_m=pid_m*BLOCK_SIZE_M;

    var start_n=pid_n*BLOCK_SIZE_N;

    var offs_am=oracle.code.triton.Triton.add(start_m,oracle.code.triton.Triton.arange(0,BLOCK_SIZE_M));

    var offs_bn=oracle.code.triton.Triton.add(start_n,oracle.code.triton.Triton.arange(0,BLOCK_SIZE_N));

    offs_am=offs_am;

    offs_bn=offs_bn;

    offs_am=java.util.Objects.requireNonNull(java.util.Objects.requireNonNull(offs_am));

    offs_bn=java.util.Objects.requireNonNull(java.util.Objects.requireNonNull(offs_bn));

    var offs_k=oracle.code.triton.Triton.arange(0,BLOCK_SIZE_K);

    var a_ptrs=oracle.code.triton.Triton.add(a_ptr,(oracle.code.triton.Triton.add(oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_am,1),stride_am),oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_k,0),stride_ak))));

    var b_ptrs=oracle.code.triton.Triton.add(b_ptr,(oracle.code.triton.Triton.add(oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_k,1),stride_bk),oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_bn,0),stride_bn))));

    var accumulator=oracle.code.triton.Triton.zeros(float.class,BLOCK_SIZE_M, BLOCK_SIZE_N);

    for (var k = 0; k < oracle.code.triton.Triton.cdiv(K,BLOCK_SIZE_K); k++) {

    var a=a_ptrs;

    var b=b_ptrs;

    accumulator=oracle.code.triton.Triton.dot(a,b);

    a_ptrs=oracle.code.triton.Triton.add(a_ptrs,BLOCK_SIZE_K*stride_ak);

    b_ptrs=oracle.code.triton.Triton.add(b_ptrs,BLOCK_SIZE_K*stride_bk);

    }Number c=0;

    if(java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))) {

    c=accumulator;

    }

    var offs_cm=oracle.code.triton.Triton.add(pid_m*BLOCK_SIZE_M,oracle.code.triton.Triton.arange(0,BLOCK_SIZE_M));

    var offs_cn=oracle.code.triton.Triton.add(pid_n*BLOCK_SIZE_N,oracle.code.triton.Triton.arange(0,BLOCK_SIZE_N));

    var c_ptrs=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(c_ptr,oracle.code.triton.Triton.mul(stride_cm,oracle.code.triton.Triton.expand(offs_cm,1))),oracle.code.triton.Triton.mul(stride_cn,oracle.code.triton.Triton.expand(offs_cn,0)));

    var c_mask=oracle.code.triton.Triton.and((oracle.code.triton.Triton.compare(oracle.code.triton.Triton.expand(offs_cm,1),M,oracle.code.triton.Triton.CompareKind.LessThan)),(oracle.code.triton.Triton.compare(oracle.code.triton.Triton.expand(offs_cn,0),N,oracle.code.triton.Triton.CompareKind.LessThan)));

    oracle.code.triton.Triton.store(c_ptrs,c,c_mask);
  }

  @Reflect
  public Object ComputeTileAndPid(int tile_id, int num_pid_in_group, int num_pid_m,
      int GROUP_SIZE_M, int NUM_SMS) {


    tile_id=tile_id+NUM_SMS;

    var group_id=tile_id/num_pid_in_group;

    var first_pid_m=group_id*GROUP_SIZE_M;

    var group_size_m=Math.min(num_pid_m-first_pid_m,GROUP_SIZE_M);

    var pid_m=first_pid_m+(tile_id%group_size_m);

    var pid_n=(tile_id%num_pid_in_group)/group_size_m;

    return null;
  }

  /**
   * @param c_ptr 
   * @param K 
   * @param stride_ak 
   * @param stride_bn 
   * @param stride_cn 
   */
  @Reflect
  public void matmulKernelPersistent(Ptr a_ptr, Ptr b_ptr, Ptr c_ptr, int M, int N, int K,
      int stride_am, int stride_ak, int stride_bk, int stride_bn, int stride_cm, int stride_cn,
      @Constant int BLOCK_SIZE_M, @Constant int BLOCK_SIZE_N, @Constant int BLOCK_SIZE_K,
      @Constant int GROUP_SIZE_M, @Constant int NUM_SMS) {


    var start_pid=oracle.code.triton.Triton.programId(0);

    var num_pid_m=oracle.code.triton.Triton.cdiv(M,BLOCK_SIZE_M);

    var num_pid_n=oracle.code.triton.Triton.cdiv(N,BLOCK_SIZE_N);

    var k_tiles=oracle.code.triton.Triton.cdiv(K,BLOCK_SIZE_K);

    var num_tiles=num_pid_m*num_pid_n;

    var tiles_per_SM=num_tiles/NUM_SMS;

    if(start_pid<num_tiles%NUM_SMS) {

    tiles_per_SM=oracle.code.triton.Triton.add(tiles_per_SM,1);

    }

    var tile_id=start_pid-NUM_SMS;

    var ki=-1;

    var offs_k_for_mask=oracle.code.triton.Triton.arange(0,BLOCK_SIZE_K);

    var num_pid_in_group=GROUP_SIZE_M*num_pid_n;

    var pid_m=0;

    var pid_n=0;

    var offs_am=oracle.code.triton.Triton.arange(0,BLOCK_SIZE_M);

    var offs_bn=oracle.code.triton.Triton.arange(0,BLOCK_SIZE_N);

    var accumulator=oracle.code.triton.Triton.zeros(float.class,BLOCK_SIZE_M, BLOCK_SIZE_N);

    for (var __ = 0; __ < k_tiles*tiles_per_SM; __++) {

    ki=0;
    int start_m=0;
    int start_n=0;

    if(java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))) {

    var __tupleValue6=(Object[])(ComputeTileAndPid(tile_id,num_pid_in_group,num_pid_m,GROUP_SIZE_M,NUM_SMS));tile_id=__tupleValue6.length>0?__tupleValue6[0]:null;pid_m=__tupleValue6.length>1&&__tupleValue6[1] instanceof Number?((Number)__tupleValue6[1]).intValue():0;pid_n=__tupleValue6.length>2&&__tupleValue6[2] instanceof Number?((Number)__tupleValue6[2]).intValue():0;

    start_m=pid_m*BLOCK_SIZE_M;

    start_n=pid_n*BLOCK_SIZE_N;

    offs_am=oracle.code.triton.Triton.add(start_m,oracle.code.triton.Triton.arange(0,BLOCK_SIZE_M));

    offs_bn=oracle.code.triton.Triton.add(start_n,oracle.code.triton.Triton.arange(0,BLOCK_SIZE_N));

    offs_am=offs_am;

    offs_bn=offs_bn;

    offs_am=java.util.Objects.requireNonNull(java.util.Objects.requireNonNull(offs_am));

    offs_bn=java.util.Objects.requireNonNull(java.util.Objects.requireNonNull(offs_bn));

    }

    var offs_k=oracle.code.triton.Triton.add(ki*BLOCK_SIZE_K,oracle.code.triton.Triton.arange(0,BLOCK_SIZE_K));

    var a_ptrs=oracle.code.triton.Triton.add(a_ptr,(oracle.code.triton.Triton.add(oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_am,1),stride_am),oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_k,0),stride_ak))));

    var b_ptrs=oracle.code.triton.Triton.add(b_ptr,(oracle.code.triton.Triton.add(oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_k,1),stride_bk),oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_bn,0),stride_bn))));

    var a=a_ptrs;

    var b=b_ptrs;

    accumulator=oracle.code.triton.Triton.dot(a,b);
    Number offs_cm=0;
    Number offs_cn=0;
    Number c_ptrs=0;
    Number c_mask=0;
    Number c=0;

    if(java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))) {

    offs_cm=oracle.code.triton.Triton.add(pid_m*BLOCK_SIZE_M,oracle.code.triton.Triton.arange(0,BLOCK_SIZE_M));

    offs_cn=oracle.code.triton.Triton.add(pid_n*BLOCK_SIZE_N,oracle.code.triton.Triton.arange(0,BLOCK_SIZE_N));

    c_ptrs=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(c_ptr,stride_cm*oracle.code.triton.Triton.expand(offs_cm,1)),stride_cn*oracle.code.triton.Triton.expand(offs_cn,0));

    c_mask=oracle.code.triton.Triton.and((oracle.code.triton.Triton.expand(offs_cm,1)<M),(oracle.code.triton.Triton.expand(offs_cn,0)<N));

    if(java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))) {

    c=accumulator;

    }

    oracle.code.triton.Triton.store(c_ptrs,c,c_mask);

    accumulator=oracle.code.triton.Triton.zeros(float.class,BLOCK_SIZE_M, BLOCK_SIZE_N);

    }

    }
  }

  /**
   * @param c_desc_ptr 
   * @param K 
   */
  @Reflect
  public void matmulKernelTmaPersistent(Ptr a_desc_ptr, Ptr b_desc_ptr, Ptr c_desc_ptr, int M,
      int N, int K, @Constant int BLOCK_SIZE_M, @Constant int BLOCK_SIZE_N,
      @Constant int BLOCK_SIZE_K, @Constant int GROUP_SIZE_M, @Constant int FP8_OUTPUT,
      @Constant int EPILOGUE_SUBTILE, @Constant int NUM_SMS) {

    //

    var dtype=0;

    var start_pid=oracle.code.triton.Triton.programId(0);

    var num_pid_m=oracle.code.triton.Triton.cdiv(M,BLOCK_SIZE_M);

    var num_pid_n=oracle.code.triton.Triton.cdiv(N,BLOCK_SIZE_N);

    var k_tiles=oracle.code.triton.Triton.cdiv(K,BLOCK_SIZE_K);

    var num_tiles=num_pid_m*num_pid_n;

    var tiles_per_SM=num_tiles/NUM_SMS;

    if(start_pid<num_tiles%NUM_SMS) {

    tiles_per_SM=oracle.code.triton.Triton.add(tiles_per_SM,1);

    }

    var tile_id=start_pid-NUM_SMS;

    // tile_id_c is used in the epilogue to break the dependency between
    // the prologue and the epilogue
    var tile_id_c=start_pid-NUM_SMS;

    var ki=-1;

    var offs_am=0;

    var offs_bn=0;

    var num_pid_in_group=GROUP_SIZE_M*num_pid_n;

    var accumulator=oracle.code.triton.Triton.zeros(float.class,BLOCK_SIZE_M, BLOCK_SIZE_N);

    for (var __ = 0; __ < k_tiles*tiles_per_SM; __++) {
    // Epilogue subtiling is a technique to break our computation and stores into multiple pieces

    ki=0;

    if(java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))) {

    var __tupleValue7=(Object[])(ComputeTileAndPid(tile_id,num_pid_in_group,num_pid_m,GROUP_SIZE_M,NUM_SMS));tile_id=__tupleValue7.length>0?__tupleValue7[0]:null;var pid_m=__tupleValue7.length>1?__tupleValue7[1]:null;var pid_n=__tupleValue7.length>2?__tupleValue7[2]:null;

    offs_am=0;

    offs_bn=0;

    }

    var offs_k=ki*BLOCK_SIZE_K;

    var a=oracle.code.triton.Triton.zeros(float.class,1);

    var b=oracle.code.triton.Triton.zeros(float.class,1);

    accumulator=oracle.code.triton.Triton.dot(a,b.T);
    Number offs_am_c=0;
    Number offs_bn_c=0;
    Number acc=0;
    Number c0=0;
    Number c1=0;

    if(java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))) {
    // Epilogue subtiling is a technique to break our computation and stores into multiple pieces

    var __tupleValue8=(Object[])(ComputeTileAndPid(tile_id_c,num_pid_in_group,num_pid_m,GROUP_SIZE_M,NUM_SMS));tile_id_c=__tupleValue8.length>0?__tupleValue8[0]:null;var pid_m=__tupleValue8.length>1?__tupleValue8[1]:null;var pid_n=__tupleValue8.length>2?__tupleValue8[2]:null;

    offs_am_c=oracle.code.triton.Triton.mul(pid_m,BLOCK_SIZE_M);

    offs_bn_c=oracle.code.triton.Triton.mul(pid_n,BLOCK_SIZE_N);

    if(true) {

    acc=accumulator;

    acc=acc;

    var __tupleValue9=(Object[])(new Object[]{acc,acc});var acc0=__tupleValue9.length>0?__tupleValue9[0]:null;var acc1=__tupleValue9.length>1?__tupleValue9[1]:null;

    c0=acc0(dtype);

    java.util.Objects.requireNonNull(c_desc_ptr);

    c1=acc1(dtype);

    java.util.Objects.requireNonNull(c_desc_ptr);

    }

    accumulator=oracle.code.triton.Triton.zeros(float.class,BLOCK_SIZE_M, BLOCK_SIZE_N);

    }

    }
  }

  /**
   * @param c_ptr 
   * @param K 
   */
  @Reflect
  public void matmulKernelDescriptorPersistent(Ptr a_ptr, Ptr b_ptr, Ptr c_ptr, int M, int N, int K,
      @Constant int BLOCK_SIZE_M, @Constant int BLOCK_SIZE_N, @Constant int BLOCK_SIZE_K,
      @Constant int GROUP_SIZE_M, @Constant int EPILOGUE_SUBTILE, @Constant int NUM_SMS) {

    //

    var dtype=0;

    var start_pid=oracle.code.triton.Triton.programId(0);

    var num_pid_m=oracle.code.triton.Triton.cdiv(M,BLOCK_SIZE_M);

    var num_pid_n=oracle.code.triton.Triton.cdiv(N,BLOCK_SIZE_N);

    var k_tiles=oracle.code.triton.Triton.cdiv(K,BLOCK_SIZE_K);

    var num_tiles=num_pid_m*num_pid_n;

    var a_desc=a_ptr;

    var b_desc=b_ptr;

    var c_desc=c_ptr;

    var tiles_per_SM=num_tiles/NUM_SMS;

    if(start_pid<num_tiles%NUM_SMS) {

    tiles_per_SM=oracle.code.triton.Triton.add(tiles_per_SM,1);

    }

    var tile_id=start_pid-NUM_SMS;

    var tile_id_c=start_pid-NUM_SMS;

    var ki=-1;

    var offs_am=0;

    var offs_bn=0;

    var num_pid_in_group=GROUP_SIZE_M*num_pid_n;

    var accumulator=oracle.code.triton.Triton.zeros(float.class,BLOCK_SIZE_M, BLOCK_SIZE_N);

    for (var __ = 0; __ < k_tiles*tiles_per_SM; __++) {

    ki=0;

    if(java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))) {

    var __tupleValue10=(Object[])(ComputeTileAndPid(tile_id,num_pid_in_group,num_pid_m,GROUP_SIZE_M,NUM_SMS));tile_id=__tupleValue10.length>0?__tupleValue10[0]:null;var pid_m=__tupleValue10.length>1?__tupleValue10[1]:null;var pid_n=__tupleValue10.length>2?__tupleValue10[2]:null;

    offs_am=0;

    offs_bn=0;

    }

    var offs_k=ki*BLOCK_SIZE_K;

    var a=0;

    var b=0;

    accumulator=oracle.code.triton.Triton.dot(a,b.T);
    Number offs_cm=0;
    Number offs_cn=0;
    Number acc=0;
    Number c0=0;
    Number c1=0;
    Number c=0;

    if(java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))) {

    var __tupleValue11=(Object[])(ComputeTileAndPid(tile_id_c,num_pid_in_group,num_pid_m,GROUP_SIZE_M,NUM_SMS));tile_id_c=__tupleValue11.length>0?__tupleValue11[0]:null;var pid_m=__tupleValue11.length>1?__tupleValue11[1]:null;var pid_n=__tupleValue11.length>2?__tupleValue11[2]:null;

    offs_cm=oracle.code.triton.Triton.mul(pid_m,BLOCK_SIZE_M);

    offs_cn=oracle.code.triton.Triton.mul(pid_n,BLOCK_SIZE_N);

    if(true) {

    acc=accumulator;

    acc=acc;

    var __tupleValue12=(Object[])(new Object[]{acc,acc});var acc0=__tupleValue12.length>0?__tupleValue12[0]:null;var acc1=__tupleValue12.length>1?__tupleValue12[1]:null;

    c0=acc0(dtype);

    java.util.Objects.requireNonNull(0);

    c1=acc1(dtype);

    java.util.Objects.requireNonNull(0);

    }

    accumulator=oracle.code.triton.Triton.zeros(float.class,BLOCK_SIZE_M, BLOCK_SIZE_N);

    }

    }
  }
}
