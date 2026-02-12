// """
// Fused Attention
// ===============
//
// This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
//
// Credits: OpenAI kernel team
//
// Extra Credits:
//
// * Original flash attention paper (https://arxiv.org/abs/2205.14135)
// * Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
//
// """
//
package org.triton4j.triton.test;

import static oracle.code.triton.Triton.*;

import java.lang.Object;
import jdk.incubator.code.Reflect;
import oracle.code.triton.Constant;
import oracle.code.triton.Ptr;

public class FusedAttention {
  /**
   * @param q 
   * @param V_block_ptr 
   * @param qk_scale 
   */
  @Reflect
  public Object AttnFwdInner(int acc, int l_i, int m_i, int q, Ptr K_block_ptr, Ptr V_block_ptr,
      int start_m, int qk_scale, @Constant int BLOCK_M, @Constant int HEAD_DIM,
      @Constant int BLOCK_N, @Constant int STAGE, @Constant int offs_m, @Constant int offs_n,
      @Constant int N_CTX, @Constant int fp8_v) {

    // range of values handled by this stage

    if(java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))) {

    var lo=0;var hi=start_m * BLOCK_M;

    }

    K_block_ptr=K_block_ptr;

    V_block_ptr=V_block_ptr;

    for (var start_n = lo; start_n < hi; start_n += BLOCK_N) {
    // -- compute qk ----

    start_n=java.util.Objects.requireNonNull(start_n);

    // -- compute qk ----
    var k=K_block_ptr;

    var qk=oracle.code.triton.Triton.dot(q,k);
    Number mask=0;
    Number m_ij=0;

    if(java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))) {

    mask=oracle.code.triton.Triton.compare(oracle.code.triton.Triton.expand(offs_m,1),(oracle.code.triton.Triton.add(start_n,oracle.code.triton.Triton.expand(offs_n,0))),oracle.code.triton.Triton.CompareKind.GreaterThanOrEqual);

    qk=oracle.code.triton.Triton.add(oracle.code.triton.Triton.mul(qk,qk_scale),0);

    m_ij=oracle.code.triton.Triton.maximum(m_i,oracle.code.triton.Triton.max(qk,1));

    qk=oracle.code.triton.Triton.sub(qk,oracle.code.triton.Triton.expand(m_ij,1));

    }

    var p=exp2(qk);

    var l_ij=oracle.code.triton.Triton.sum(p,1);

    // -- update m_i and l_i
    var alpha=exp2(m_i-m_ij);

    l_i=0;

    // -- update output accumulator --
    acc=0;

    // update acc
    var v=V_block_ptr;

    if(true) {

    p=p;

    }

    acc=0;

    // update m_i and l_i
    m_i=m_ij;

    V_block_ptr=V_block_ptr;

    K_block_ptr=K_block_ptr;

    }
    return null;
  }

  /**
   * @param q 
   * @param desc_v 
   * @param qk_scale 
   */
  @Reflect
  public Object AttnFwdInnerTma(int acc, int l_i, int m_i, int q, int desc_k, int desc_v,
      int offset_y, @Constant int dtype, int start_m, int qk_scale, @Constant int BLOCK_M,
      @Constant int HEAD_DIM, @Constant int BLOCK_N, @Constant int STAGE, @Constant int offs_m,
      @Constant int offs_n, @Constant int N_CTX) {

    // range of values handled by this stage

    if(java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))) {

    var lo=0;var hi=start_m * BLOCK_M;

    }

    var offsetkv_y=offset_y+lo;

    for (var start_n = lo; start_n < hi; start_n += BLOCK_N) {
    // -- compute qk ----

    start_n=java.util.Objects.requireNonNull(start_n);

    // -- compute qk ----
    var k=0;

    var qk=oracle.code.triton.Triton.dot(q,k);
    Number mask=0;
    Number m_ij=0;

    if(java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))) {

    mask=oracle.code.triton.Triton.compare(oracle.code.triton.Triton.expand(offs_m,1),(oracle.code.triton.Triton.add(start_n,oracle.code.triton.Triton.expand(offs_n,0))),oracle.code.triton.Triton.CompareKind.GreaterThanOrEqual);

    qk=oracle.code.triton.Triton.add(oracle.code.triton.Triton.mul(qk,qk_scale),0);

    m_ij=oracle.code.triton.Triton.maximum(m_i,oracle.code.triton.Triton.max(qk,1));

    qk=oracle.code.triton.Triton.sub(qk,oracle.code.triton.Triton.expand(m_ij,1));

    }

    var p=exp2(qk);

    var l_ij=oracle.code.triton.Triton.sum(p,1);

    // -- update m_i and l_i
    var alpha=exp2(m_i-m_ij);

    l_i=0;

    // -- update output accumulator --
    acc=0;

    // update acc
    var v=oracle.code.triton.Triton.zeros(float.class,1);

    p=p(dtype);

    // note that this non transposed v for FP8 is only supported on Blackwell
    acc=0;

    // update m_i and l_i
    m_i=m_ij;

    offsetkv_y=oracle.code.triton.Triton.add(offsetkv_y,BLOCK_N);

    }
    return null;
  }

  /**
   * @param Out 
   * @param stride_qk 
   * @param stride_kk 
   * @param stride_vn 
   * @param stride_on 
   * @param N_CTX 
   */
  @Reflect
  public void AttnFwd(int Q, int K, int V, int sm_scale, int M, int Out, int stride_qz,
      int stride_qh, int stride_qm, int stride_qk, int stride_kz, int stride_kh, int stride_kn,
      int stride_kk, int stride_vz, int stride_vh, int stride_vk, int stride_vn, int stride_oz,
      int stride_oh, int stride_om, int stride_on, int Z, int H, int N_CTX, @Constant int HEAD_DIM,
      @Constant int BLOCK_M, @Constant int BLOCK_N, @Constant int STAGE) {

    // block pointers

    Integer.valueOf(0);

    var start_m=oracle.code.triton.Triton.programId(0);

    var off_hz=oracle.code.triton.Triton.programId(1);

    var off_z=off_hz/H;

    var off_h=off_hz%H;

    var qvk_offset=oracle.code.triton.Triton.add(oracle.code.triton.Triton.mul(off_z,stride_qz),oracle.code.triton.Triton.mul(off_h,stride_qh));

    // block pointers
    var Q_block_ptr=base=Q + qvk_offset;

    var v_order=0;

    var V_block_ptr=base=V + qvk_offset;

    var K_block_ptr=base=K + qvk_offset;

    var O_block_ptr=base=Out + qvk_offset;

    // initialize offsets
    var offs_m=oracle.code.triton.Triton.add(start_m*BLOCK_M,oracle.code.triton.Triton.arange(0,BLOCK_M));

    var offs_n=oracle.code.triton.Triton.arange(0,BLOCK_N);

    // initialize pointer to m and l
    var m_i=oracle.code.triton.Triton.sub(oracle.code.triton.Triton.zeros(float.class,0),Float.parseFloat("inf"));

    var l_i=oracle.code.triton.Triton.add(oracle.code.triton.Triton.zeros(float.class,0),1.0);

    var acc=oracle.code.triton.Triton.zeros(float.class,0);

    // load scales
    var qk_scale=sm_scale;

    // 1/log(2)
    qk_scale=oracle.code.triton.Triton.mul(qk_scale,1.44269504);

    // load q: it will stay in SRAM throughout
    var q=Q_block_ptr;

    if(java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))) {
    //

    //
    //
    //
    //
    var __tupleValue0=(Object[])(AttnFwdInner(acc,l_i,m_i,q,K_block_ptr,V_block_ptr,start_m,qk_scale,BLOCK_M,HEAD_DIM,BLOCK_N,4-STAGE,offs_m,offs_n,N_CTX,oracle.code.triton.Triton.compare(0,0,oracle.code.triton.Triton.CompareKind.Equal)));acc=__tupleValue0.length>0?__tupleValue0[0]:null;l_i=__tupleValue0.length>1?__tupleValue0[1]:null;m_i=__tupleValue0.length>2?__tupleValue0[2]:null;

    }

    if(java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))) {
    // barrier makes it easier for compielr to schedule the

    //
    //
    //
    //
    var __tupleValue1=(Object[])(AttnFwdInner(acc,l_i,m_i,q,K_block_ptr,V_block_ptr,start_m,qk_scale,BLOCK_M,HEAD_DIM,BLOCK_N,2,offs_m,offs_n,N_CTX,oracle.code.triton.Triton.compare(0,0,oracle.code.triton.Triton.CompareKind.Equal)));acc=__tupleValue1.length>0?__tupleValue1[0]:null;l_i=__tupleValue1.length>1?__tupleValue1[1]:null;m_i=__tupleValue1.length>2?__tupleValue1[2]:null;

    }

    m_i=oracle.code.triton.Triton.add(m_i,log2(l_i));

    acc=oracle.code.triton.Triton.div(acc,oracle.code.triton.Triton.expand(l_i,1));

    var m_ptrs=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(M,off_hz*N_CTX),offs_m);

    java.util.Objects.requireNonNull(m_ptrs);

    java.util.Objects.requireNonNull(O_block_ptr);
  }

  /**
   * @param M 
   * @param N_CTX 
   */
  @Reflect
  public void AttnFwdTma(int sm_scale, int M, int Z, int H, int desc_q, int desc_k, int desc_v,
      int desc_o, int N_CTX, @Constant int HEAD_DIM, @Constant int BLOCK_M, @Constant int BLOCK_N,
      @Constant int FP8_OUTPUT, @Constant int STAGE) {

    // initialize offsets

    var dtype=0;

    Integer.valueOf(0);

    var start_m=oracle.code.triton.Triton.programId(0);

    var off_hz=oracle.code.triton.Triton.programId(1);

    var off_z=off_hz/H;

    var off_h=off_hz%H;

    var offset_y=off_z+off_h*N_CTX;

    var qo_offset_y=offset_y+start_m*BLOCK_M;

    // initialize offsets
    var offs_m=oracle.code.triton.Triton.add(start_m*BLOCK_M,oracle.code.triton.Triton.arange(0,BLOCK_M));

    var offs_n=oracle.code.triton.Triton.arange(0,BLOCK_N);

    // initialize pointer to m and l
    var m_i=oracle.code.triton.Triton.sub(oracle.code.triton.Triton.zeros(float.class,0),Float.parseFloat("inf"));

    var l_i=oracle.code.triton.Triton.add(oracle.code.triton.Triton.zeros(float.class,0),1.0);

    var acc=oracle.code.triton.Triton.zeros(float.class,0);

    // load scales
    var qk_scale=sm_scale;

    // 1/log(2)
    qk_scale=oracle.code.triton.Triton.mul(qk_scale,1.44269504);

    // load q: it will stay in SRAM throughout
    var q=oracle.code.triton.Triton.zeros(float.class,1);

    if(java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))) {
    //

    //
    //
    //
    //
    //
    var __tupleValue2=(Object[])(AttnFwdInnerTma(acc,l_i,m_i,q,desc_k,desc_v,offset_y,dtype,start_m,qk_scale,BLOCK_M,HEAD_DIM,BLOCK_N,4-STAGE,offs_m,offs_n,N_CTX));acc=__tupleValue2.length>0?__tupleValue2[0]:null;l_i=__tupleValue2.length>1?__tupleValue2[1]:null;m_i=__tupleValue2.length>2?__tupleValue2[2]:null;

    }

    if(java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))) {
    // barrier makes it easier for compielr to schedule the

    //
    //
    //
    //
    //
    var __tupleValue3=(Object[])(AttnFwdInnerTma(acc,l_i,m_i,q,desc_k,desc_v,offset_y,dtype,start_m,qk_scale,BLOCK_M,HEAD_DIM,BLOCK_N,2,offs_m,offs_n,N_CTX));acc=__tupleValue3.length>0?__tupleValue3[0]:null;l_i=__tupleValue3.length>1?__tupleValue3[1]:null;m_i=__tupleValue3.length>2?__tupleValue3[2]:null;

    }

    m_i=oracle.code.triton.Triton.add(m_i,log2(l_i));

    acc=oracle.code.triton.Triton.div(acc,oracle.code.triton.Triton.expand(l_i,1));

    var m_ptrs=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(M,off_hz*N_CTX),offs_m);

    java.util.Objects.requireNonNull(m_ptrs);

    java.util.Objects.requireNonNull(desc_o);
  }

  /**
   * @param DO 
   * @param Delta 
   * @param N_CTX 
   */
  @Reflect
  public void AttnBwdPreprocess(int O, int DO, int Delta, int Z, int H, int N_CTX,
      @Constant int BLOCK_M, @Constant int HEAD_DIM) {

    // load

    var off_m=oracle.code.triton.Triton.add(oracle.code.triton.Triton.programId(0)*BLOCK_M,oracle.code.triton.Triton.arange(0,BLOCK_M));

    var off_hz=oracle.code.triton.Triton.programId(1);

    var off_n=oracle.code.triton.Triton.arange(0,HEAD_DIM);

    // load
    var o=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(O,off_hz*HEAD_DIM*N_CTX),oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(off_m,1),HEAD_DIM)),oracle.code.triton.Triton.expand(off_n,0));

    var do_=oracle.code.triton.Triton.zeros(float.class,1);

    var delta=oracle.code.triton.Triton.sum(oracle.code.triton.Triton.mul(o,do_),1);

    // write-back
    java.util.Objects.requireNonNull(oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(Delta,off_hz*N_CTX),off_m));
  }

  /**
   * @param dv 
   * @param sm_scale 
   * @param DO 
   * @param D  shared by Q/K/V/DO.
   * @param stride_d 
   * @param num_steps 
   */
  @Reflect
  public Object AttnBwdDkdv(int dk, int dv, int Q, int k, int v, int sm_scale, int DO, int M, int D,
      int stride_tok, int stride_d, int H, int N_CTX, @Constant int BLOCK_M1,
      @Constant int BLOCK_N1, @Constant int HEAD_DIM, int start_n, int start_m, int num_steps,
      @Constant int MASK) {

    // BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.

    var offs_m=oracle.code.triton.Triton.add(start_m,oracle.code.triton.Triton.arange(0,BLOCK_M1));

    var offs_n=oracle.code.triton.Triton.add(start_n,oracle.code.triton.Triton.arange(0,BLOCK_N1));

    var offs_k=oracle.code.triton.Triton.arange(0,HEAD_DIM);

    var qT_ptrs=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(Q,oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_m,0),stride_tok)),oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_k,1),stride_d));

    var do_ptrs=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(DO,oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_m,1),stride_tok)),oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_k,0),stride_d));

    // BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    Integer.valueOf(0);

    var curr_m=start_m;

    var step_m=BLOCK_M1;

    for (var blk_idx = 0; blk_idx < num_steps; blk_idx++) {
    // Load m before computing qk to reduce pipeline stall.

    var qT=qT_ptrs;

    // Load m before computing qk to reduce pipeline stall.
    offs_m=oracle.code.triton.Triton.add(curr_m,oracle.code.triton.Triton.arange(0,BLOCK_M1));

    var m=oracle.code.triton.Triton.add(M,offs_m);

    var qkT=oracle.code.triton.Triton.dot(k,qT);

    var pT=exp2(oracle.code.triton.Triton.sub(qkT,oracle.code.triton.Triton.expand(m,0)));
    Number mask=0;

    if(true) {

    mask=(oracle.code.triton.Triton.compare(oracle.code.triton.Triton.expand(offs_m,0),oracle.code.triton.Triton.expand(offs_n,1),oracle.code.triton.Triton.CompareKind.GreaterThanOrEqual));

    pT=pT;

    }

    var do_=do_ptrs;

    // Compute dV.
    var ppT=pT;

    ppT=ppT;

    dv=dv+oracle.code.triton.Triton.dot(ppT,do_);

    // D (= delta) is pre-divided by ds_scale.
    var Di=oracle.code.triton.Triton.add(D,offs_m);

    // Compute dP and dS.
    var dpT=oracle.code.triton.Triton.zeros(float.class,1);

    var dsT=oracle.code.triton.Triton.mul(pT,(oracle.code.triton.Triton.sub(dpT,oracle.code.triton.Triton.expand(Di,0))));

    dsT=dsT;

    dk=dk+oracle.code.triton.Triton.dot(dsT,oracle.code.triton.Triton.trans(qT));

    // Increment pointers.
    curr_m=oracle.code.triton.Triton.add(curr_m,step_m);

    qT_ptrs=oracle.code.triton.Triton.add(qT_ptrs,oracle.code.triton.Triton.mul(step_m,stride_tok));

    do_ptrs=oracle.code.triton.Triton.add(do_ptrs,oracle.code.triton.Triton.mul(step_m,stride_tok));

    }
    return null;
  }

  /**
   * @param V 
   * @param D  shared by Q/K/V/DO.
   * @param stride_d 
   * @param N_CTX 
   * @param num_steps 
   */
  @Reflect
  public Object AttnBwdDq(int dq, int q, int K, int V, int do_, int m, int D, int stride_tok,
      int stride_d, int H, int N_CTX, @Constant int BLOCK_M2, @Constant int BLOCK_N2,
      @Constant int HEAD_DIM, int start_m, int start_n, int num_steps, @Constant int MASK) {

    // D (= delta) is pre-divided by ds_scale.

    var offs_m=oracle.code.triton.Triton.add(start_m,oracle.code.triton.Triton.arange(0,BLOCK_M2));

    var offs_n=oracle.code.triton.Triton.add(start_n,oracle.code.triton.Triton.arange(0,BLOCK_N2));

    var offs_k=oracle.code.triton.Triton.arange(0,HEAD_DIM);

    var kT_ptrs=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(K,oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_n,0),stride_tok)),oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_k,1),stride_d));

    var vT_ptrs=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(V,oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_n,0),stride_tok)),oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_k,1),stride_d));

    // D (= delta) is pre-divided by ds_scale.
    var Di=oracle.code.triton.Triton.add(D,offs_m);

    // BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    Integer.valueOf(0);

    var curr_n=start_n;

    var step_n=BLOCK_N2;

    for (var blk_idx = 0; blk_idx < num_steps; blk_idx++) {
    // Autoregressive masking.

    var kT=kT_ptrs;

    var vT=vT_ptrs;

    var qk=oracle.code.triton.Triton.dot(q,kT);

    var p=exp2(oracle.code.triton.Triton.sub(qk,m));
    Number mask=0;

    if(true) {
    // Compute dP and dS.

    offs_n=oracle.code.triton.Triton.add(curr_n,oracle.code.triton.Triton.arange(0,BLOCK_N2));

    mask=(oracle.code.triton.Triton.compare(oracle.code.triton.Triton.expand(offs_m,1),oracle.code.triton.Triton.expand(offs_n,0),oracle.code.triton.Triton.CompareKind.GreaterThanOrEqual));

    p=p;

    }

    var dp=oracle.code.triton.Triton.zeros(float.class,1);

    var ds=oracle.code.triton.Triton.mul(p,(oracle.code.triton.Triton.sub(dp,oracle.code.triton.Triton.expand(Di,1))));

    ds=ds;

    // Compute dQ.
    // NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
    dq=dq+oracle.code.triton.Triton.dot(ds,oracle.code.triton.Triton.trans(kT));

    // Increment pointers.
    curr_n=oracle.code.triton.Triton.add(curr_n,step_n);

    kT_ptrs=oracle.code.triton.Triton.add(kT_ptrs,oracle.code.triton.Triton.mul(step_n,stride_tok));

    vT_ptrs=oracle.code.triton.Triton.add(vT_ptrs,oracle.code.triton.Triton.mul(step_n,stride_tok));

    }
    return dq;
  }

  /**
   * @param sm_scale 
   * @param DO 
   * @param DV 
   * @param D  shared by Q/K/V/DO.
   * @param stride_d 
   * @param N_CTX 
   */
  @Reflect
  public void AttnBwd(int Q, int K, int V, int sm_scale, int DO, int DQ, int DK, int DV, int M,
      int D, int stride_z, int stride_h, int stride_tok, int stride_d, int H, int N_CTX,
      @Constant int BLOCK_M1, @Constant int BLOCK_N1, @Constant int BLOCK_M2,
      @Constant int BLOCK_N2, @Constant int BLK_SLICE_FACTOR, @Constant int HEAD_DIM) {

    // = ln(2)

    // = ln(2)
    var LN2=0.6931471824645996;

    var bhid=oracle.code.triton.Triton.programId(2);

    var off_chz=oracle.code.triton.Triton.zeros(float.class,1);

    var adj=oracle.code.triton.Triton.zeros(float.class,1);

    var pid=oracle.code.triton.Triton.programId(0);

    // offset pointers for batch/head
    Q=Q+adj;

    K=K+adj;

    V=V+adj;

    DO=DO+adj;

    DQ=DQ+adj;

    DK=DK+adj;

    DV=DV+adj;

    M=M+off_chz;

    D=D+off_chz;

    // load scales
    var offs_k=oracle.code.triton.Triton.arange(0,HEAD_DIM);

    var start_n=pid*BLOCK_N1;

    var start_m=start_n;

    var MASK_BLOCK_M1=BLOCK_M1/BLK_SLICE_FACTOR;

    var offs_n=oracle.code.triton.Triton.add(start_n,oracle.code.triton.Triton.arange(0,BLOCK_N1));

    var dv=oracle.code.triton.Triton.zeros(float.class,0);

    var dk=oracle.code.triton.Triton.zeros(float.class,0);

    // load K and V: they stay in SRAM throughout the inner loop.
    var k=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(K,oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_n,1),stride_tok)),oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_k,0),stride_d));

    var v=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(V,oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_n,1),stride_tok)),oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_k,0),stride_d));

    var num_steps=BLOCK_N1/MASK_BLOCK_M1;

    //
    //
    //
    //
    //
    //
    //
    //
    //
    var __tupleValue4=(Object[])(AttnBwdDkdv(dk,dv,Q,k,v,sm_scale,DO,M,D,stride_tok,stride_d,H,N_CTX,MASK_BLOCK_M1,BLOCK_N1,HEAD_DIM,start_n,start_m,num_steps,True));dk=__tupleValue4.length>0?__tupleValue4[0]:null;dv=__tupleValue4.length>1?__tupleValue4[1]:null;

    start_m=oracle.code.triton.Triton.add(start_m,num_steps*MASK_BLOCK_M1);

    num_steps=oracle.code.triton.Triton.div((oracle.code.triton.Triton.sub(N_CTX,start_m)),BLOCK_M1);

    // Compute dK and dV for non-masked blocks.
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    var __tupleValue5=(Object[])(AttnBwdDkdv(dk,dv,Q,k,v,sm_scale,DO,M,D,stride_tok,stride_d,H,N_CTX,BLOCK_M1,BLOCK_N1,HEAD_DIM,start_n,start_m,num_steps,False));dk=__tupleValue5.length>0?__tupleValue5[0]:null;dv=__tupleValue5.length>1?__tupleValue5[1]:null;

    var dv_ptrs=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(DV,oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_n,1),stride_tok)),oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_k,0),stride_d));

    java.util.Objects.requireNonNull(dv_ptrs);

    // Write back dK.
    dk=oracle.code.triton.Triton.mul(dk,sm_scale);

    var dk_ptrs=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(DK,oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_n,1),stride_tok)),oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_k,0),stride_d));

    java.util.Objects.requireNonNull(dk_ptrs);

    // THIS BLOCK DOES DQ:
    start_m=pid*BLOCK_M2;

    var end_n=oracle.code.triton.Triton.add(start_m,BLOCK_M2);

    var MASK_BLOCK_N2=BLOCK_N2/BLK_SLICE_FACTOR;

    var offs_m=oracle.code.triton.Triton.add(start_m,oracle.code.triton.Triton.arange(0,BLOCK_M2));

    var q=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(Q,oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_m,1),stride_tok)),oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_k,0),stride_d));

    var dq=oracle.code.triton.Triton.zeros(float.class,0);

    var do_=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(DO,oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_m,1),stride_tok)),oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_k,0),stride_d));

    var m=oracle.code.triton.Triton.add(M,offs_m);

    m=oracle.code.triton.Triton.expand(m,1);

    // Compute dQ for masked (diagonal) blocks.
    // NOTE: This code scans each row of QK^T backward (from right to left,
    // but inside each call to _attn_bwd_dq, from left to right), but that's
    // not due to anything important.  I just wanted to reuse the loop
    // structure for dK & dV above as much as possible.
    num_steps=BLOCK_M2/MASK_BLOCK_N2;

    //
    //
    //
    //
    //
    //
    //
    dq=AttnBwdDq(dq,q,K,V,do_,m,D,stride_tok,stride_d,H,N_CTX,BLOCK_M2,MASK_BLOCK_N2,HEAD_DIM,start_m,oracle.code.triton.Triton.sub(end_n,num_steps*MASK_BLOCK_N2),num_steps,True);

    end_n=oracle.code.triton.Triton.sub(end_n,num_steps*MASK_BLOCK_N2);

    // stage 2
    num_steps=oracle.code.triton.Triton.div(end_n,BLOCK_N2);

    //
    //
    //
    //
    //
    //
    //
    dq=AttnBwdDq(dq,q,K,V,do_,m,D,stride_tok,stride_d,H,N_CTX,BLOCK_M2,BLOCK_N2,HEAD_DIM,start_m,oracle.code.triton.Triton.sub(end_n,num_steps*BLOCK_N2),num_steps,False);

    // Write back dQ.
    var dq_ptrs=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(DQ,oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_m,1),stride_tok)),oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(offs_k,0),stride_d));

    dq=oracle.code.triton.Triton.mul(dq,LN2);

    java.util.Objects.requireNonNull(dq_ptrs);
  }
}
