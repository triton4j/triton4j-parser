// GraalPy parity test input.
package org.triton4j.codegen.graalpy;

import static oracle.code.triton.Triton.*;

import java.lang.Object;
import jdk.incubator.code.Reflect;
import oracle.code.triton.Constant;

public class ParityKernels {
  @Reflect
  public Object addOne(@Constant int x) {


    return x+1;
  }

  @Reflect
  public Object mulAdd(@Constant int x, @Constant int y, @Constant int z) {


    return x*y+z;
  }
}
