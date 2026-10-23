# Variable (runtime-index) permutes, compress and partition.
#
# LLVM has no portable "permute by a runtime index vector" operation, and its
# portable compress (`llvm.masked.compressstore`, `llvm.experimental.vector.compress`)
# is expanded lane by lane on targets without a native compress instruction,
# e.g. AVX2. The functions below use target-specific instructions where
# `SIMD.Targets` allows it, and portable LLVM IR otherwise; both give identical
# results.

const PermuteTypes = Union{IntegerTypes, FloatingTypes}

_llvmbits(t::String) = parse(Int, t[2:end])        # "i32" -> 32

# IR converting `<N x it> %src` to `<N x i\$to> %dst` (zext or trunc).
function _ir_convert_index(N, it, to, src, dst)
    from = _llvmbits(it)
    from < to && return "%$dst = zext <$N x $it> %$src to <$N x i$to>"
    from > to && return "%$dst = trunc <$N x $it> %$src to <$N x i$to>"
    return "%$dst = add <$N x $it> %$src, zeroinitializer"
end

_ir_vec(xs, t) = "<" * join(("$t $x" for x in xs), ", ") * ">"

# Portable: lane i = extractelement(v, idx[i]).
function _ir_permute_generic(N, t, it)
    lines = String[]
    for i in 0:(N - 1)
        push!(lines, "%i$i = extractelement <$N x $it> %idx, i32 $i")
        push!(lines, "%e$i = extractelement <$N x $t> %v, $it %i$i")
    end
    prev = "poison"
    for i in 0:(N - 1)
        push!(lines, "%r$i = insertelement <$N x $t> $prev, $t %e$i, i32 $i")
        prev = "%r$i"
    end
    return """
        define <$N x $t> @entry(<$N x $t> %v, <$N x $it> %idx) #0 {
        top:
            $(join(lines, "\n    "))
            ret <$N x $t> $prev
        }
        attributes #0 = { alwaysinline }"""
end

# AVX2 `vpermd` on a 256-bit vector of 32-bit (N = 8) or 64-bit (N = 4) lanes.
function _ir_permute_avx2(N, t, it)
    if N == 8
        conv = _ir_convert_index(8, it, 32, "idx", "i")
    else  # 64-bit lanes: index j becomes the 32-bit pair (2j, 2j + 1)
        conv = join([_ir_convert_index(4, it, 32, "idx", "i4"),
                     "%i2 = shl <4 x i32> %i4, " * _ir_vec((1, 1, 1, 1), "i32"),
                     "%dup = shufflevector <4 x i32> %i2, <4 x i32> poison, <8 x i32> " *
                         _ir_vec((0, 0, 1, 1, 2, 2, 3, 3), "i32"),
                     "%i = or <8 x i32> %dup, " * _ir_vec((0, 1, 0, 1, 0, 1, 0, 1), "i32")],
                    "\n    ")
    end
    return """
        declare <8 x i32> @llvm.x86.avx2.permd(<8 x i32>, <8 x i32>)
        define <$N x $t> @entry(<$N x $t> %v, <$N x $it> %idx) #0 {
        top:
            $conv
            %vb = bitcast <$N x $t> %v to <8 x i32>
            %r = call <8 x i32> @llvm.x86.avx2.permd(<8 x i32> %vb, <8 x i32> %i)
            %rb = bitcast <8 x i32> %r to <$N x $t>
            ret <$N x $t> %rb
        }
        attributes #0 = { alwaysinline }"""
end

# SSSE3 `pshufb` on a 128-bit vector of any lane width: lane index j becomes the
# byte indices s*j .. s*j + s - 1 (s = bytes per lane).
function _ir_permute_pshufb(N, t, it, s)
    if s == 1
        conv = _ir_convert_index(16, it, 8, "idx", "b")
    else
        conv = join([_ir_convert_index(N, it, 8, "idx", "i8"),
                     "%m = mul <$N x i8> %i8, " * _ir_vec(fill(s, N), "i8"),
                     "%rep = shufflevector <$N x i8> %m, <$N x i8> poison, <16 x i32> " *
                         _ir_vec([j for j in 0:(N - 1) for _ in 1:s], "i32"),
                     "%b = add <16 x i8> %rep, " * _ir_vec([k for _ in 1:N for k in 0:(s - 1)], "i8")],
                    "\n    ")
    end
    return """
        declare <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8>, <16 x i8>)
        define <$N x $t> @entry(<$N x $t> %v, <$N x $it> %idx) #0 {
        top:
            $conv
            %vb = bitcast <$N x $t> %v to <16 x i8>
            %r = call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %vb, <16 x i8> %b)
            %rb = bitcast <16 x i8> %r to <$N x $t>
            ret <$N x $t> %rb
        }
        attributes #0 = { alwaysinline }"""
end

@generated function _vpermute(v::Vec{N,T}, idx::Vec{N,I}, ::Val{P}) where {N, T<:PermuteTypes, I<:IntegerTypes, P}
    t = Intrinsics.d[T]
    it = Intrinsics.d[I]
    s = sizeof(T)
    ir = if P === :auto && N * s == 32 && s in (4, 8) && Targets.has_feature(:avx2)
        _ir_permute_avx2(N, t, it)
    elseif P === :auto && N * s == 16 && Targets.has_feature(:ssse3)
        _ir_permute_pshufb(N, t, it, s)
    else
        _ir_permute_generic(N, t, it)
    end
    return quote
        $(Expr(:meta, :inline))
        Vec(Base.llvmcall(($ir, "entry"), LVec{N,T}, Tuple{LVec{N,T}, LVec{N,I}}, v.data, idx.data))
    end
end

"""
    vpermute(v::Vec{N,T}, idx::Vec{N,<:Integer}) -> Vec{N,T}

Permute the lanes of `v` by the runtime indices `idx`: lane `i` of the result is
`v[idx[i] + 1]`. Indices are 0-based, like [`shufflevector`](@ref), but need not
be compile-time constants. Out-of-range indices give unspecified lanes.

Lowers to one `vpermd` for 256-bit vectors of 32- or 64-bit lanes on AVX2, to one
`pshufb` for 128-bit vectors on SSSE3, and to portable LLVM IR otherwise (see
[`SIMD.Targets`](@ref)).

# Example
```jldoctest
julia> vpermute(Vec{4,Int}((10, 20, 30, 40)), Vec{4,Int}((3, 0, 0, 2)))
<4 x Int64>[40, 10, 10, 30]
```
"""
@inline vpermute(v::Vec{N,T}, idx::Vec{N,<:IntegerTypes}) where {N, T<:PermuteTypes} =
    _vpermute(v, idx, Val(:auto))
