// """
// Libdevice (`tl.extra.libdevice`) function
// ==============================
// Triton can invoke a custom function from an external library.
// In this example, we will use the `libdevice` library to apply `asin` on a tensor.
//
// Please refer to `CUDA libdevice-users-guide <https://docs.nvidia.com/cuda/libdevice-users-guide/index.html>`_ and/or `HIP device-lib source code <https://github.com/ROCm/llvm-project/tree/amd-staging/amd/device-libs/ocml/src>`_ regarding the semantics of all available libdevice functions.
//
// In `libdevice.py`, we try to aggregate functions with the same computation but different data types together.
// For example, both `__nv_asin` and `__nv_asinf` calculate the principal value of the arc sine of the input, but `__nv_asin` operates on `double` and `__nv_asinf` operates on `float`.
// Triton automatically selects the correct underlying device function to invoke based on input and output types.
// """
//
package org.triton4j.samples.generated;

import static oracle.code.triton.Triton.*;

import jdk.incubator.code.Reflect;
import oracle.code.triton.Constant;
import oracle.code.triton.Ptr;

public class K07ExternFunctions {
  @Reflect
  public void asinKernel(Ptr x_ptr, Ptr y_ptr, int n_elements, @Constant int BLOCK_SIZE) {

    // %%

    var pid=oracle.code.triton.Triton.programId(0);

    var block_start=pid*BLOCK_SIZE;

    var offsets=oracle.code.triton.Triton.add(block_start,oracle.code.triton.Triton.arange(0,BLOCK_SIZE));

    var mask=oracle.code.triton.Triton.compare(offsets,n_elements,oracle.code.triton.Triton.CompareKind.LessThan);

    var x=oracle.code.triton.Triton.add(x_ptr,offsets);

    x=x;

    oracle.code.triton.Triton.store(oracle.code.triton.Triton.add(y_ptr,offsets),x,mask);
  }
}
