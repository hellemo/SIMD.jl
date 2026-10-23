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

# --- Compress and partition ---------------------------------------------------

# For N in (2, 4, 8): entry m + 1 lists the lanes selected by bitmask m in
# order, then the unselected lanes in order (0-based lane indices).
function _make_partition_table(N)
    return [Vec{N,UInt8}(Tuple(UInt8.(vcat([j for j in 0:(N - 1) if (m >> j) & 1 == 1],
                                           [j for j in 0:(N - 1) if (m >> j) & 1 == 0]))))
            for m in 0:((1 << N) - 1)]
end
const _PARTITION_TABLE_2 = _make_partition_table(2)
const _PARTITION_TABLE_4 = _make_partition_table(4)
const _PARTITION_TABLE_8 = _make_partition_table(8)
@inline _partition_table(::Val{2}) = _PARTITION_TABLE_2
@inline _partition_table(::Val{4}) = _PARTITION_TABLE_4
@inline _partition_table(::Val{8}) = _PARTITION_TABLE_8

const _HAS_VECTOR_COMPRESS = Base.libllvm_version >= v"19"   # llvm.experimental.vector.compress

# Whether vpermute has a single-instruction lowering for this shape.
_fast_permute(N, s) = (N * s == 32 && s in (4, 8) && Targets.has_feature(:avx2)) ||
                      (N * s == 16 && Targets.has_feature(:ssse3))

# 4 x 64-bit lanes as 8 x 32-bit index pairs, so that AVX2 needs no index
# arithmetic per call: lane j becomes (2j, 2j + 1).
const _PARTITION_PAIRS_4 = [Vec{8,UInt32}(Tuple(UInt32(2 * Int(t[l]) + h) for l in 1:4 for h in 0:1))
                            for t in _PARTITION_TABLE_4]

@generated function _table_partition(v::Vec{N,T}, mask::Vec{N,Bool}, ::Val{P}) where {N,T,P}
    if P === :auto && N == 4 && sizeof(T) == 8 && Targets.has_feature(:avx2)
        return quote
            $(Expr(:meta, :inline))
            idx = @inbounds _PARTITION_PAIRS_4[Int(bitmask(mask)) + 1]
            reinterpret(Vec{4,T}, _vpermute(reinterpret(Vec{8,UInt32}, v), idx, Val(:auto)))
        end
    end
    return quote
        $(Expr(:meta, :inline))
        idx = @inbounds _partition_table(Val(N))[Int(bitmask(mask)) + 1]
        _vpermute(v, idx, Val(P))
    end
end

function _ir_vector_compress(N, t, sfx)
    return """
        declare <$N x $t> @llvm.experimental.vector.compress.$sfx(<$N x $t>, <$N x i1>, <$N x $t>)
        define <$N x $t> @entry(<$N x $t> %v, <$N x i8> %m8) #0 {
        top:
            %m = trunc <$N x i8> %m8 to <$N x i1>
            %r = call <$N x $t> @llvm.experimental.vector.compress.$sfx(<$N x $t> %v, <$N x i1> %m, <$N x $t> poison)
            ret <$N x $t> %r
        }
        attributes #0 = { alwaysinline }"""
end

@inline function _scalar_compress(v::Vec{N,T}, mask::Vec{N,Bool}) where {N,T}
    r = v
    j = 1
    @inbounds for i in 1:N
        r = Base.setindex(r, v[i], min(j, N))
        j += mask[i]
    end
    return r
end

@generated function _vcompress(v::Vec{N,T}, mask::Vec{N,Bool}, ::Val{P}) where {N, T<:PermuteTypes, P}
    s = sizeof(T)
    table = N in (2, 4, 8)
    avx512 = Targets.has_feature(:avx512f) && Targets.has_feature(:avx512vl)
    if N == 1
        # Nothing to move. (LLVM 20/23 also crash when legalizing a 1-lane
        # llvm.experimental.vector.compress, which is scalarized.)
        return :($(Expr(:meta, :inline)); v)
    elseif table && P === :auto && _fast_permute(N, s) && !avx512
        return :($(Expr(:meta, :inline)); _table_partition(v, mask, Val(P)))
    elseif _HAS_VECTOR_COMPRESS
        ir = _ir_vector_compress(N, Intrinsics.d[T], Intrinsics.suffix(N, T))
        return quote
            $(Expr(:meta, :inline))
            Vec(Base.llvmcall(($ir, "entry"), LVec{N,T}, Tuple{LVec{N,T}, LVec{N,Bool}}, v.data, mask.data))
        end
    elseif table
        return :($(Expr(:meta, :inline)); _table_partition(v, mask, Val(P)))
    else
        return :($(Expr(:meta, :inline)); _scalar_compress(v, mask))
    end
end

"""
    vcompress(v::Vec{N,T}, mask::Vec{N,Bool}) -> Vec{N,T}

Move the lanes of `v` selected by `mask` to the front, keeping their order. The
remaining `N - count(mask)` lanes are unspecified.

Lowers to a table lookup plus one [`vpermute`](@ref) for 2, 4 or 8 lanes where
that is a single `vpermd`/`pshufb` (AVX2/SSSE3), to LLVM's
`llvm.experimental.vector.compress` otherwise (LLVM ≥ 19; a native `vpcompress`
on AVX-512), and to portable code before LLVM 19. See also [`vpartition`](@ref).

# Example
```jldoctest
julia> v = Vec{4,Int}((10, 20, 30, 40));

julia> vcompress(v, v > 15)[1:3]
(20, 30, 40)
```
"""
@inline vcompress(v::Vec{N,T}, mask::Vec{N,Bool}) where {N, T<:PermuteTypes} =
    _vcompress(v, mask, Val(:auto))

@inline function _vpartition(v::Vec{N,T}, mask::Vec{N,Bool}, ::Val{P}) where {N,T,P}
    k = count(mask)
    if N == 2 || N == 4 || N == 8
        return _table_partition(v, mask, Val(P)), k
    end
    # Selected lanes first, then the unselected ones shifted to start at lane k.
    lo = _vcompress(v, mask, Val(P))
    hi = _vcompress(v, !mask, Val(P))
    lane = Vec{N,Int32}(ntuple(i -> Int32(i - 1), Val(N)))
    shifted = _vpermute(hi, vifelse(lane >= Int32(k), lane - Int32(k), lane), Val(P))
    return vifelse(lane < Int32(k), lo, shifted), k
end

"""
    vpartition(v::Vec{N,T}, mask::Vec{N,Bool}) -> (Vec{N,T}, Int)

Stable two-way partition of the lanes of `v`: the lanes selected by `mask` first,
then the others, each group in its original order, together with
`count(mask)`. This is the vectorized partition step of quicksort: storing the
result once at each end of the output keeps both groups.

For 2, 4 or 8 lanes this is one table lookup and one [`vpermute`](@ref), i.e. a
single `vpermd`/`pshufb` on AVX2/SSSE3.

# Example
```jldoctest
julia> v = Vec{4,Int}((10, 40, 20, 30));

julia> vpartition(v, v < 25)
(Vec{4, Int64}((10, 20, 40, 30)), 2)
```
"""
@inline vpartition(v::Vec{N,T}, mask::Vec{N,Bool}) where {N, T<:PermuteTypes} =
    _vpartition(v, mask, Val(:auto))
