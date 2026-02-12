// """
// Low-Memory Dropout
// ==================
//
// In this tutorial, you will write a memory-efficient implementation of dropout whose state
// will be composed of a single int32 seed. This differs from more traditional implementations of dropout,
// whose state is generally composed of a bit mask tensor of the same shape as the input.
//
// In doing so, you will learn about:
//
// * The limitations of naive implementations of Dropout with PyTorch.
//
// * Parallel pseudo-random number generation in Triton.
//
// """
//
package org.triton4j.samples.generated;

import static oracle.code.triton.Triton.*;

import jdk.incubator.code.Reflect;
import oracle.code.triton.Constant;
import oracle.code.triton.Ptr;

public class K04LowMemoryDropout {
  /**
   * @param x_ptr  pointer to the input
   * @param x_keep_ptr  pointer to a mask of 0s and 1s
   * @param output_ptr  pointer to the output
   * @param n_elements  number of elements in the `x` tensor
   * @param p  probability that an element of `x` is changed to zero
   */
  @Reflect
  public void Dropout(Ptr x_ptr, Ptr x_keep_ptr, Ptr output_ptr, int n_elements, int p,
      @Constant int BLOCK_SIZE) {

    // Load data

    var pid=oracle.code.triton.Triton.programId(0);

    var block_start=pid*BLOCK_SIZE;

    var offsets=oracle.code.triton.Triton.add(block_start,oracle.code.triton.Triton.arange(0,BLOCK_SIZE));

    var mask=oracle.code.triton.Triton.compare(offsets,n_elements,oracle.code.triton.Triton.CompareKind.LessThan);

    // Load data
    var x=oracle.code.triton.Triton.add(x_ptr,offsets);

    var x_keep=oracle.code.triton.Triton.add(x_keep_ptr,offsets);

    // The line below is the crucial part, described in the paragraph above!
    var output=oracle.code.triton.Triton.div(x,(1-p));

    // Write-back output
    oracle.code.triton.Triton.store(oracle.code.triton.Triton.add(output_ptr,offsets),output,mask);
  }

  @Reflect
  public void SeededDropout(Ptr x_ptr, Ptr output_ptr, int n_elements, int p, int seed,
      @Constant int BLOCK_SIZE) {

    // compute memory offsets of elements handled by this instance

    var pid=oracle.code.triton.Triton.programId(0);

    var block_start=pid*BLOCK_SIZE;

    var offsets=oracle.code.triton.Triton.add(block_start,oracle.code.triton.Triton.arange(0,BLOCK_SIZE));

    // load data from x
    var mask=oracle.code.triton.Triton.compare(offsets,n_elements,oracle.code.triton.Triton.CompareKind.LessThan);

    var x=oracle.code.triton.Triton.add(x_ptr,offsets);

    // randomly prune it
    var random=offsets;

    var x_keep=oracle.code.triton.Triton.compare(random,p,oracle.code.triton.Triton.CompareKind.GreaterThan);

    // write-back
    var output=oracle.code.triton.Triton.div(x,(1-p));

    oracle.code.triton.Triton.store(oracle.code.triton.Triton.add(output_ptr,offsets),output,mask);
  }
}
