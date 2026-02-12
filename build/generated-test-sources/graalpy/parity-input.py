def _jit(fn):
    return fn

class _Triton:
    jit = staticmethod(_jit)

class _TL:
    constexpr = int

triton = _Triton()
tl = _TL()

@triton.jit
def add_one(x: tl.constexpr):
    return x + 1

@triton.jit
def mul_add(x: tl.constexpr, y: tl.constexpr, z: tl.constexpr):
    return x * y + z
