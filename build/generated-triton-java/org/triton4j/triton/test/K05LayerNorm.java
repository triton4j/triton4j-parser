// """
// Layer Normalization
// ====================
// In this tutorial, you will write a high-performance layer normalization
// kernel that runs faster than the PyTorch implementation.
//
// In doing so, you will learn about:
//
// * Implementing backward pass in Triton.
//
// * Implementing parallel reduction in Triton.
//
// """
//
package org.triton4j.triton.test;

import static oracle.code.triton.Triton.*;

import jdk.incubator.code.Reflect;
import oracle.code.triton.Constant;
import oracle.code.triton.Tensor;

public class K05LayerNorm {
  /**
   * @param X  pointer to the input
   * @param Y  pointer to the output
   * @param W  pointer to the weights
   * @param B  pointer to the biases
   * @param Mean  pointer to the mean
   * @param Rstd  pointer to the 1/std
   * @param stride  how much to increase the pointer when moving by 1 row
   * @param N  number of columns in X
   * @param eps  epsilon to avoid division by zero
   */
  @Reflect
  public void LayerNormFwdFused(Tensor X, Tensor Y, Tensor W, Tensor B, Tensor Mean, Tensor Rstd,
      int stride, int N, int eps, @Constant int BLOCK_SIZE) {

    // Map the program id to the row of X and Y it should compute.

    var row=oracle.code.triton.Triton.programId(0);

    Y=oracle.code.triton.Triton.add(Y,row*stride);

    X=oracle.code.triton.Triton.add(X,row*stride);

    // Compute mean
    var mean=0;

    var _mean=oracle.code.triton.Triton.zeros(float.class,0);

    for (var off = 0; off < N; off += BLOCK_SIZE) {

    var cols=oracle.code.triton.Triton.add(off,oracle.code.triton.Triton.arange(0,BLOCK_SIZE));

    var a=oracle.code.triton.Triton.zeros(float.class,1);

    _mean=oracle.code.triton.Triton.add(_mean,a);

    }
    mean=0;

    // Compute variance
    var _var=oracle.code.triton.Triton.zeros(float.class,0);

    for (var off = 0; off < N; off += BLOCK_SIZE) {

    var cols=oracle.code.triton.Triton.add(off,oracle.code.triton.Triton.arange(0,BLOCK_SIZE));

    var x=oracle.code.triton.Triton.zeros(float.class,1);

    x=oracle.code.triton.Triton.sub(x,mean);

    _var=oracle.code.triton.Triton.add(_var,oracle.code.triton.Triton.mul(x,x));

    }
    var var=oracle.code.triton.Triton.div(oracle.code.triton.Triton.sum(_var,0),N);

    var rstd=oracle.code.triton.Triton.div(1,oracle.code.triton.Triton.add(var,eps));

    // Write mean / rstd
    java.util.Objects.requireNonNull(oracle.code.triton.Triton.add(Mean,row));

    java.util.Objects.requireNonNull(oracle.code.triton.Triton.add(Rstd,row));

    for (var off = 0; off < N; off += BLOCK_SIZE) {
    // Write output

    var cols=oracle.code.triton.Triton.add(off,oracle.code.triton.Triton.arange(0,BLOCK_SIZE));

    var mask=oracle.code.triton.Triton.compare(cols,N,oracle.code.triton.Triton.CompareKind.LessThan);

    var w=oracle.code.triton.Triton.add(W,cols);

    var b=oracle.code.triton.Triton.add(B,cols);

    var x=oracle.code.triton.Triton.zeros(float.class,1);

    var x_hat=oracle.code.triton.Triton.mul((oracle.code.triton.Triton.sub(x,mean)),rstd);

    var y=oracle.code.triton.Triton.add(oracle.code.triton.Triton.mul(x_hat,w),b);

    // Write output
    oracle.code.triton.Triton.store(oracle.code.triton.Triton.add(Y,cols),y,mask);

    }
  }

  /**
   * @param DX  pointer to the input gradient
   * @param DY  pointer to the output gradient
   * @param DW  pointer to the partial sum of weights gradient
   * @param DB  pointer to the partial sum of biases gradient
   * @param X  pointer to the input
   * @param W  pointer to the weights
   * @param Mean  pointer to the mean
   * @param Rstd  pointer to the 1/std
   * @param Lock  pointer to the lock
   * @param stride  how much to increase the pointer when moving by 1 row
   * @param N  number of columns in X
   */
  @Reflect
  public void LayerNormBwdDxFused(Tensor DX, Tensor DY, Tensor DW, Tensor DB, Tensor X, Tensor W,
      Tensor Mean, Tensor Rstd, Tensor Lock, int stride, int N, @Constant int GROUP_SIZE_M,
      @Constant int BLOCK_SIZE_N) {

    // Map the program id to the elements of X, DX, and DY it should compute.

    var row=oracle.code.triton.Triton.programId(0);

    var cols=oracle.code.triton.Triton.arange(0,BLOCK_SIZE_N);

    var mask=oracle.code.triton.Triton.compare(cols,N,oracle.code.triton.Triton.CompareKind.LessThan);

    X=oracle.code.triton.Triton.add(X,row*stride);

    DY=oracle.code.triton.Triton.add(DY,row*stride);

    DX=oracle.code.triton.Triton.add(DX,row*stride);

    // Offset locks and weights/biases gradient pointer for parallel reduction
    var lock_id=row%GROUP_SIZE_M;

    Lock=oracle.code.triton.Triton.add(Lock,lock_id);

    var Count=oracle.code.triton.Triton.add(Lock,GROUP_SIZE_M);

    DW=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(DW,lock_id*N),cols);

    DB=oracle.code.triton.Triton.add(oracle.code.triton.Triton.add(DB,lock_id*N),cols);

    // Load data to SRAM
    var x=oracle.code.triton.Triton.zeros(float.class,1);

    var dy=oracle.code.triton.Triton.zeros(float.class,1);

    var w=oracle.code.triton.Triton.zeros(float.class,1);

    var mean=oracle.code.triton.Triton.add(Mean,row);

    var rstd=oracle.code.triton.Triton.add(Rstd,row);

    // Compute dx
    var xhat=oracle.code.triton.Triton.mul((oracle.code.triton.Triton.sub(x,mean)),rstd);

    var wdy=oracle.code.triton.Triton.mul(w,dy);

    xhat=xhat;

    wdy=wdy;

    var c1=oracle.code.triton.Triton.div(oracle.code.triton.Triton.sum(oracle.code.triton.Triton.mul(xhat,wdy),0),N);

    var c2=oracle.code.triton.Triton.div(oracle.code.triton.Triton.sum(wdy,0),N);

    var dx=oracle.code.triton.Triton.mul((oracle.code.triton.Triton.sub(wdy,(oracle.code.triton.Triton.add(oracle.code.triton.Triton.mul(xhat,c1),c2)))),rstd);

    // Write dx
    oracle.code.triton.Triton.store(oracle.code.triton.Triton.add(DX,cols),dx,mask);

    // Accumulate partial sums for dw/db
    var partial_dw=oracle.code.triton.Triton.zeros(float.class,1);

    var partial_db=oracle.code.triton.Triton.zeros(float.class,1);

    while(java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))) {




    }
    var count=Count;

    if(java.util.Objects.nonNull(oracle.code.triton.Triton.zeros(float.class,1))) {

    Integer.valueOf(0);

    }

    oracle.code.triton.Triton.store(DW,partial_dw,mask);

    oracle.code.triton.Triton.store(DB,partial_db,mask);

    // Release the lock
    Integer.valueOf(0);
  }

  /**
   * @param DW  pointer to the partial sum of weights gradient
   * @param DB  pointer to the partial sum of biases gradient
   * @param FINAL_DW  pointer to the weights gradient
   * @param FINAL_DB  pointer to the biases gradient
   * @param M  GROUP_SIZE_M
   * @param N  number of columns
   */
  @Reflect
  public void LayerNormBwdDwdb(Tensor DW, Tensor DB, Tensor FINAL_DW, Tensor FINAL_DB, int M, int N,
      @Constant int BLOCK_SIZE_M, @Constant int BLOCK_SIZE_N) {

    // Map the program id to the elements of DW and DB it should compute.

    var pid=oracle.code.triton.Triton.programId(0);

    var cols=oracle.code.triton.Triton.add(pid*BLOCK_SIZE_N,oracle.code.triton.Triton.arange(0,BLOCK_SIZE_N));

    var dw=oracle.code.triton.Triton.zeros(float.class,BLOCK_SIZE_M, BLOCK_SIZE_N);

    var db=oracle.code.triton.Triton.zeros(float.class,BLOCK_SIZE_M, BLOCK_SIZE_N);

    for (var i = 0; i < M; i += BLOCK_SIZE_M) {
    // Write the final sum to the output.

    var rows=oracle.code.triton.Triton.add(i,oracle.code.triton.Triton.arange(0,BLOCK_SIZE_M));

    var mask=oracle.code.triton.Triton.and((oracle.code.triton.Triton.compare(oracle.code.triton.Triton.expand(rows,1),M,oracle.code.triton.Triton.CompareKind.LessThan)),(oracle.code.triton.Triton.compare(oracle.code.triton.Triton.expand(cols,0),N,oracle.code.triton.Triton.CompareKind.LessThan)));

    var offs=oracle.code.triton.Triton.add(oracle.code.triton.Triton.mul(oracle.code.triton.Triton.expand(rows,1),N),oracle.code.triton.Triton.expand(cols,0));

    dw=oracle.code.triton.Triton.add(dw,oracle.code.triton.Triton.add(DW,offs));

    db=oracle.code.triton.Triton.add(db,oracle.code.triton.Triton.add(DB,offs));

    }
    var sum_dw=oracle.code.triton.Triton.sum(dw,0);

    var sum_db=oracle.code.triton.Triton.sum(db,0);

    oracle.code.triton.Triton.store(oracle.code.triton.Triton.add(FINAL_DW,cols),sum_dw,oracle.code.triton.Triton.compare(cols,N,oracle.code.triton.Triton.CompareKind.LessThan));

    oracle.code.triton.Triton.store(oracle.code.triton.Triton.add(FINAL_DB,cols),sum_db,oracle.code.triton.Triton.compare(cols,N,oracle.code.triton.Triton.CompareKind.LessThan));
  }
}
