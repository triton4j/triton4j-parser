// """
// Vector Addition
// ===============
//
// In this tutorial, you will write a simple vector addition using Triton.
//
// In doing so, you will learn about:
//
// * The basic programming model of Triton.
//
// * The `triton.jit` decorator, which is used to define Triton kernels.
//
// * The best practices for validating and benchmarking your custom ops against native reference implementations.
//
// """
//
package org.triton4j.triton.test;

import static oracle.code.triton.Triton.*;

import jdk.incubator.code.Reflect;
import oracle.code.triton.Constant;
import oracle.code.triton.Ptr;

public class VectorAdd {
  /**
   * @param x_ptr  *Pointer* to first input vector.
   * @param y_ptr  *Pointer* to second input vector.
   * @param output_ptr  *Pointer* to output vector.
   * @param n_elements  Size of the vector.
   */
  @Reflect
  public void addKernel(Ptr x_ptr, Ptr y_ptr, Ptr output_ptr, int n_elements,
      @Constant int BLOCK_SIZE) {

    // There are multiple 'programs' processing different data. We identify which program

    // We use a 1D launch grid so axis is 0.
    var pid=oracle.code.triton.Triton.programId(0);

    // This program will process inputs that are offset from the initial data.
    // For instance, if you had a vector of length 256 and block_size of 64, the programs
    // would each access the elements [0:64, 64:128, 128:192, 192:256].
    // Note that offsets is a list of pointers:
    var block_start=pid*BLOCK_SIZE;

    var offsets=oracle.code.triton.Triton.add(block_start,oracle.code.triton.Triton.arange(0,BLOCK_SIZE));

    // Create a mask to guard memory operations against out-of-bounds accesses.
    var mask=oracle.code.triton.Triton.compare(offsets,n_elements,oracle.code.triton.Triton.CompareKind.LessThan);

    // Load x and y from DRAM, masking out any extra elements in case the input is not a
    // multiple of the block size.
    var x=oracle.code.triton.Triton.add(x_ptr,offsets);

    var y=oracle.code.triton.Triton.add(y_ptr,offsets);

    var output=oracle.code.triton.Triton.add(x,y);

    // Write x + y back to DRAM.
    oracle.code.triton.Triton.store(oracle.code.triton.Triton.add(output_ptr,offsets),output,mask);
  }
}
