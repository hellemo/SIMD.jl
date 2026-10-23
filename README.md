# SIMD

Explicit SIMD vectorization in Julia

[![Code Coverage](https://codecov.io/gh/eschnett/SIMD.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/eschnett/SIMD.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3355421.svg)](https://doi.org/10.5281/zenodo.3355421)
[![PkgEval](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/S/SIMD.svg)](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/S/SIMD.html)

| Julia   | CI |
| ------- | -- |
| v1      | [![CI](https://github.com/eschnett/SIMD.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/eschnett/SIMD.jl/actions/workflows/ci.yml) |
| nightly | [![CI (Julia nightly)](https://github.com/eschnett/SIMD.jl/actions/workflows/ci_julia_nightly.yml/badge.svg)](https://github.com/eschnett/SIMD.jl/actions/workflows/ci_julia_nightly.yml) |

## Overview

This package allows programmers to explicitly SIMD-vectorize their
Julia code. Ideally, the compiler (Julia and LLVM) would be able to do
this automatically, especially for straightforwardly written code. In
practice, this does not always work (for a variety of reasons), and
the programmer is often left with uncertainty as to whether the code
was actually vectorized. It is usually necessary to look at the
generated machine code to determine whether the compiler actually
vectorized the code.

By exposing SIMD vector types and corresponding operations, the programmer can explicitly vectorize their code. While this does not guarantee that the generated machine code is efficient, it relieves the compiler from determining whether it is legal to vectorize the code, deciding whether it is beneficial to do so, and rearranging the code to synthesize vector instructions.

Here is a simple example for a manually vectorized code that adds two arrays:
```Julia
using SIMD
function vadd!(xs::Vector{T}, ys::Vector{T}, ::Type{Vec{N,T}}) where {N, T}
    @assert length(ys) == length(xs)
    @assert length(xs) % N == 0
    lane = VecRange{N}(0)
    @inbounds for i in 1:N:length(xs)
        xs[lane + i] += ys[lane + i]
    end
end
```

To simplify this example code, the vector type that should be used (`Vec{N,T}`) is passed in explicitly as additional type argument. This routine is e.g. called as `vadd!(xs, ys, Vec{8,Float64})`.
Note that this code is not expected to outperform the standard scalar way of
doing this operation since the Julia optimizer will easily rewrite that to use
SIMD under the hood. It is merely shown as an illustration of how to load and
store data into `Vector`s using SIMD.jl

## SIMD vector operations

SIMD vectors are similar to small fixed-size arrays of "simple" types. These element types are supported:

`Bool Int{8,16,32,64,128} UInt{8,16,32,64,128} Float{16,32,64}`

The SIMD package provides the usual arithmetic and logical operations for SIMD vectors:

`+ - * / % ^ ! ~ & | $ << >> >>> == != < <= > >=`

`abs ceil copysign cos div eps exp exp2 flipsign floor fma ifelse¹ inv isfinite isinf isnan issubnormal log log10 log2 muladd rem round sign signbit sin sqrt trunc vifelse`

(Currently missing: `exponent ldexp significand`, many trigonometric functions)  
¹ Supported on Julia v1.8 and above. For compatibility with lower versions use `vifelse` instead.

These operators and functions are always applied element-wise, i.e. they are applied to each element in parallel, yielding again a SIMD vector as result. This means that e.g. multiplying two vectors yields a vector, and comparing two vectors yields a vector of booleans. This behaviour might seem strange and slightly unusual, but corresponds to the machine instructions provided by the hardware. It is also what is usually needed to vectorize loops.

The SIMD package also provides conversion operators from scalars and tuples to SIMD vectors and from SIMD vectors to tuples. Additionally, there are `getindex` and `setindex` functions to access individual vector elements.  SIMD vectors are immutable (like tuples), and `setindex` (note there is no exclamation mark at the end of the name) thus returns the modified vector.

```julia
# Create a vector where all elements are Float64(1):
xs = Vec{4,Float64}(1)

# Create a vector from a tuple, and convert it back to a tuple:
ys = Vec{4,Float32}((1,2,3,4))
ys1 = NTuple{4,Float32}(ys)
y2 = ys[2]   # getindex

# Update one element of a vector:
ys = Base.setindex(ys, 5, 3)   # cannot use ys[3] = 5
```

## Reduction operations

Reduction operations reduce a SIMD vector to a scalar. The following reduction operations are provided:

`all any maximum minimum sum prod bitmask`

Example:

```julia
# Sum over a vector
v = Vec{4,Float64}((1,2,3,4))
sum(v)
10.0

# Reduce to a bitmask
v = Vec{8,Bool}((0,1,1,0,1,1,0,1))
bitmask(v)
0xb6
```

It is also possible to use reduce with bit operations:

```julia
julia> v = Vec{4,UInt16}((1,2,3,4))
<4 x UInt16>[0x0001, 0x0002, 0x0003, 0x0004]

julia> reduce(|, v)
0x0007

julia> reduce(&, v)
0x0000
```

## Overflow operations

Overflow operations do the operation but also give back a flag that indicates
whether the result of the operation overflowed.
Note that these only work on Julia with LLVM 9 or higher (Julia 1.5 or higher):
The functions `Base.Checked.add_with_overflow`, `Base.Checked.sub_with_overflow`,
`Base.Checked.mul_with_overflow` are extended to work on `Vec`. :

```julia
julia> v = Vec{4, Int8}((40, -80, 70, -10))
<4 x Int8>[40, -80, 70, -10]

julia> Base.Checked.add_with_overflow(v, v)
(<4 x Int8>[80, 96, -116, -20], <4 x Bool>[0, 1, 1, 0])

julia> Base.Checked.add_with_overflow(Int8(-80), Int8(-80))
(96, true)

julia> Base.Checked.sub_with_overflow(v, 120)
(<4 x Int8>[-80, 56, -50, 126], <4 x Bool>[0, 1, 0, 1])

julia> Base.Checked.mul_with_overflow(v, 2)
(<4 x Int8>[80, 96, -116, -20], <4 x Bool>[0, 1, 1, 0])
```

## Saturation arithmetic

Saturation arithmetic is a version of arithmetic in which operations are limited
to a fixed range between a minimum and maximum value. If the result of an
operation is greater than the maximum value, the result is set (or “clamped”) to
this maximum. If it is below the minimum, it is clamped to this minimum.


```julia
julia> v = Vec{4, Int8}((40, -80, 70, -10))
<4 x Int8>[40, -80, 70, -10]

julia> SIMD.add_saturate(v, v)
<4 x Int8>[80, -128, 127, -20]

julia> SIMD.sub_saturate(v, 120)
<4 x Int8>[-80, -128, -50, -128]
```

## Fastmath

SIMD.jl hooks into the `@fastmath` macro so that operations in a
`@fastmath`-block sets the `fast` flag on the floating point intrinsics
that supports it operations. Compare for example the generated code for the
following two functions:

```julia
f1(a, b, c) = a * b - c * 2.0
f2(a, b, c) = @fastmath a * b - c * 2.0
V = Vec{4, Float64}
code_native(f1, Tuple{V, V, V}, debuginfo=:none)
code_native(f2, Tuple{V, V, V}, debuginfo=:none)
```

The normal caveats for using `@fastmath` naturally applies.

## Accessing arrays

When using explicit SIMD vectorization, it is convenient to allocate arrays still as arrays of scalars, not as arrays of vectors. The `vload` and `vstore` functions allow reading vectors from and writing vectors into arrays, accessing several contiguous array elements.

```Julia
arr = Vector{Float64}(undef, 100)
...
xs = vload(Vec{4,Float64}, arr, i)
...
vstore(xs, arr, i)
```
The `vload` call reads a vector of size 4 from the array, i.e. it reads `arr[i:i+3]`. Similarly, the `vstore` call writes the vector `xs` to the four array elements `arr[i:i+3]`.

When the values to be read are stored in non-contiguous locations, the `vgather` function can be used to load them into a vector (so-called gather operation).

```Julia
idx = Vec((1, 5, 6, 9))
vgather(arr, idx)
```

Likewise, storing to non-contiguous locations (scatter) can be done by the `vscatter` function.

```Julia
arr = zeros(10)
v = Vec((1.0, 2.0, 3.0, 4.0))
idx = Vec((1, 3, 4, 7))
vscatter(v, arr, idx)
```

Above `vload`, `vstore`, `vgather` and `vscatter` can be written using the indexing notation:

```Julia
i = 1
lane = VecRange{4}(0)
v = arr[lane + i]             # vload
arr[lane + i] = v             # vstore
idx = Vec((1, 3, 4, 7))
v = arr[idx]                  # vgather
arr[idx] = v                  # vscatter
```

### `@inbounds` usage

Note that `vload`, `vstore` etc, by default, check that the indices are in
bounds of the array. These boundschecks can be turned off using the `@inbounds`
macro, e.g. `@inbounds vload(V, a, i)`.  This is often crucial for good
performance.

### GPU backends

There is experimental support for using `vload`, `vstore`, `vgather` and `vscatter`
on GPU device arrays with supported backends. Support will vary from backend to
backend. Currently, only OpenCL.jl is tested in CI. OpenCL by default only support
`vload` and `vstore` on `CLDeviceArray`s. For `vgather` and `vscatter`, the
`SPV_INTEL_masked_gather_scatter` extension is required, support for which
currently requires some workarounds such as using the `:khronos` instead of the
`:llvm` backend and disabling SPIRV validation. See the tests in
[`test/opencl.jl`](test/opencl.jl) for some examples.

On GPU backends, using `@inbounds` is essential for good performance, as exceptions
have a large overhead. Otherwise - so far supported - SIMD operations should work
just like on the CPU.

## Vector shuffles

Vector shuffle is available through the `shufflevector` function.

Example:
```Julia
a = Vec{4, Int32}((1,2,3,4))
b = Vec{4, Int32}((5,6,7,8))
mask = (2,3,4,5)
shufflevector(a, b, Val(mask))
<4 x Int32>[3, 4, 5, 6]
```
The mask specifies vector elements counted across `a` and `b`,
starting at 0 to follow the LLVM convention. If you don't care about
some of the values in the result vector, you can use the symbol
`:undef`. `a` and `b` must be of the same SIMD vector type. The
result will be a SIMD vector with the same element type as `a` and `b`
and the same length as the mask. The function must be specialized on
the value of the mask, therefore the `Val()` construction in the call.

There is also a one operand version of the function:
```Julia
a = Vec{4, Int32}((1,2,3,4))
mask = (0,3,1,2)
shufflevector(a, Val(mask))
<4 x Int32>[1, 4, 2, 3]
```

## Runtime permutes, compress and partition

`shufflevector` needs the permutation at compile time. For permutations that
depend on the data, e.g. in vectorized sorting, partitioning or filtering,
there are:

- `vpermute(v, idx)`: lane `i` of the result is `v[idx[i] + 1]` (0-based
  runtime indices).
- `vcompress(v, mask)`: the lanes selected by `mask` moved to the front, in
  order; the rest unspecified.
- `vpartition(v, mask)`: the selected lanes followed by the unselected ones,
  each in order, and `count(mask)`; one vectorized quicksort partition step.
- `vstorec(v, ptr, mask)`: compress-store exactly `count(mask)` elements.
  `vstorec_overwrite(v, ptr, mask)` stores all `N` lanes of `vcompress(v, mask)`,
  which is faster when the destination has room.
- `count(mask)`: the number of `true` lanes.

```Julia
v = Vec{8,Int32}((5, 1, 7, 3, 8, 2, 6, 4))
w, k = vpartition(v, v < 5)     # ([1, 3, 2, 4, 5, 7, 8, 6], 4)

# Out-of-place partition around a pivot: the low side grows upwards from
# lo[1], the high side downwards from hi[end]. One vpartition per vector;
# each full-width store's extra lanes land where the next store will write.
lo = Vector{Int32}(undef, n + 8); hi = similar(lo); a = 1; r = length(hi)
for i in 1:8:n
    x = vload(Vec{8,Int32}, keys, i)
    w, k = vpartition(x, x < pivot)
    vstore(w, lo, a)            # lanes 1:k are the keys < pivot
    vstore(w, hi, r - 7)        # lanes k+1:8 land at hi[r-7+k:r]
    a += k; r -= 8 - k
end
# keys < pivot: lo[1:a-1]; keys ≥ pivot: hi[r+1:end]
```

LLVM's portable compress (`llvm.masked.compressstore`,
`llvm.experimental.vector.compress`) is expanded lane by lane on targets without
a compress instruction, such as AVX2, and LLVM has no portable runtime permute.
These functions therefore choose their lowering when they are compiled, using
`SIMD.Targets`:

| Target | `vpermute` | `vcompress`, `vpartition` (2, 4, 8 lanes) | `vstorec` |
|---|---|---|---|
| AVX2, 256-bit vectors of 32/64-bit lanes | `vpermd` | table of permutations + `vpermd` | compress + masked store |
| SSSE3, 128-bit vectors | `pshufb` | table + `pshufb` | compress + masked store (AVX2) |
| AVX-512 | portable | `llvm.experimental.vector.compress` (`vpcompress`) | `llvm.masked.compressstore` |
| other | portable | `llvm.experimental.vector.compress` (LLVM ≥ 19), else table | `llvm.masked.compressstore` |

Target-specific instructions are only used when Julia compiles for the host CPU
(`--cpu-target=native`, the default); set `SIMD_JL_GENERIC=1` to force the
portable lowering. On AVX2, prefer 256-bit vectors (8 × 32-bit or 4 × 64-bit
lanes): wider vectors use the slow portable compress.
`julia --project=. benchmark/partition.jl` measures the partition kernels on
the current machine; on an AVX2 laptop (Intel Core Ultra 7 268V) `vpartition`
partitions 8 × `UInt32` at 0.21 ns per key, against 0.56 for branch-free
scalar code and 4.3 for `llvm.masked.compressstore`.

## Representing SIMD vector types in Julia

In LLVM, SIMD vectors are represented via a special vector type. LLVM supports vectors of all "primitive" types, i.e. integers (including booleans), floating-point numbers, and pointers. LLVM directly provides arithmetic and logic operations (add, subtract, bit shift, select, etc.) for vector types. For example, adding two numbers is represented in LLVM as
```LLVM
%res = fadd <double> %arg1, %arg2
```
and adding two vectors looks like
```LLVM
%res = fadd <4 x double> %arg1, %arg2
```

Thus, implementing SIMD operations in Julia is in principle a straightforward application of `llvmcall`. In principle, this should work:
```Julia
function +(x::Float64x4, y::Float64x4)
    llvmcall("""
        %res = fadd <4 x double> %0, %1
        ret <4 x double> %res
        """, Float64x4, Tuple{Float64x4, Float64x4}, x, y)
end
```

The Julia representation of the datatype `Float64x4` is slightly
complex: It is an `NTuple{N,T}`, where the element type `T` is
specially marked by being wrapped in the type `Base.VecElement`:
`NTuple{4, Base.VecElement{Float64}}`. Julia implements a special rule
that translates tuples with element type `Base.VecElement` into LLVM
vectors. Other tuples are translated into LLVM arrays if all tuple
elements have the same type, otherwise into LLVM structures.

This representation has two drawbacks. First, it is rather tedious.
Second, while we want to define arithmetic operations for SIMD
vectors, we do not want to define arithmetic for Julia's tuple types
-- if we defined additional methods for generic tuples, who knows what
code would break as a result.

We thus define our own SIMD vector type `Vec{N,T}`:
```Julia
struct Vec{N,T}
    elts::NTuple{N,VecElement{T}}
end
```
