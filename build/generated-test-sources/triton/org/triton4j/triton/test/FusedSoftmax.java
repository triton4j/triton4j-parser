// """
// Fused Softmax
// =============
//
// In this tutorial, you will write a fused softmax operation that is significantly faster
// than PyTorch's native op for a particular class of matrices: those whose rows can fit in
// the GPU's SRAM.
//
// In doing so, you will learn about:
//
// * The benefits of kernel fusion for bandwidth-bound operations.
//
// * Reduction operators in Triton.
//
// """
//
package org.triton4j.triton.test;

import static oracle.code.triton.Triton.*;

import jdk.incubator.code.Reflect;
import oracle.code.triton.Constant;
import oracle.code.triton.Ptr;

public class FusedSoftmax {
  @Reflect
  public void softmaxKernel(Ptr output_ptr, Ptr input_ptr, int input_row_stride,
      int output_row_stride, int n_rows, int n_cols, @Constant int BLOCK_SIZE,
      @Constant int num_stages) {

    // starting row of the program

    var row_start=oracle.code.triton.Triton.programId(0);

    var row_step=1;

    for (var row_idx = row_start; row_idx < n_rows; row_idx += row_step) {
    // The stride represents how much we need to increase the pointer to advance 1 row

    var row_start_ptr=oracle.code.triton.Triton.add(input_ptr,oracle.code.triton.Triton.mul(row_idx,input_row_stride));

    // The block size is the next power of two greater than n_cols, so we can fit each
    // row in a single block
    var col_offsets=oracle.code.triton.Triton.arange(0,BLOCK_SIZE);

    var input_ptrs=oracle.code.triton.Triton.add(row_start_ptr,col_offsets);

    // Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    var mask=oracle.code.triton.Triton.compare(col_offsets,n_cols,oracle.code.triton.Triton.CompareKind.LessThan);

    var row=input_ptrs;

    // Subtract maximum for numerical stability
    var row_minus_max=oracle.code.triton.Triton.sub(row,oracle.code.triton.Triton.max(row,0));

    // Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    var numerator=oracle.code.triton.Triton.exp(row_minus_max);

    var denominator=oracle.code.triton.Triton.sum(numerator,0);

    var softmax_output=oracle.code.triton.Triton.div(numerator,denominator);

    // Write back output to DRAM
    var output_row_start_ptr=oracle.code.triton.Triton.add(output_ptr,oracle.code.triton.Triton.mul(row_idx,output_row_stride));

    var output_ptrs=oracle.code.triton.Triton.add(output_row_start_ptr,col_offsets);

    oracle.code.triton.Triton.store(output_ptrs,softmax_output,mask);

    }
  }
}
