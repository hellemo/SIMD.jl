# Benchmark of SIMD partition/filter kernels: scalar code vs SIMD.jl's
# compress operations, plus code generation checks. Self-contained; run from
# the repository root with
#
#     julia --project=. benchmark/partition.jl
#
# and compare with `SIMD_JL_GENERIC=1 julia --project=. benchmark/partition.jl`
# (portable lowering only). Prints a Markdown table.

using SIMD, InteractiveUtils, Printf

const KEYS = 2^20
const REPS = 20

# Scalar branch-free two-sided partition (out of place).
function partition_scalar!(lo, hi, x, pivot)
    a = 1; b = 1
    @inbounds for v in x
        c = v < pivot
        lo[a] = v; hi[b] = v
        a += c; b += !c
    end
    return a - 1
end

# Two compress stores per vector, one per side.
function partition_vstorec!(lo::Vector{T}, hi::Vector{T}, x::Vector{T}, pivot::T,
                            ::Val{N}, store::F) where {T,N,F}
    a = 1; b = 1; p = Vec{N,T}(pivot)
    GC.@preserve lo hi @inbounds for i in 1:N:length(x)
        v = vload(Vec{N,T}, x, i); m = v < p
        store(v, pointer(lo, a), m); store(v, pointer(hi, b), !m)
        k = count(m); a += k; b += N - k
    end
    return a - 1
end

# One vpartition per vector, stored at both outputs (full-width stores): the
# low side grows upwards from lo[1], the high side downwards from hi[end].
function partition_vpartition!(lo::Vector{T}, hi::Vector{T}, x::Vector{T}, pivot::T,
                               ::Val{N}) where {T,N}
    a = 1; r = length(hi); p = Vec{N,T}(pivot)
    @inbounds for i in 1:N:length(x)
        v = vload(Vec{N,T}, x, i)
        w, k = vpartition(v, v < p)
        vstore(w, lo, a); vstore(w, hi, r - N + 1)
        a += k; r -= N - k
    end
    return a - 1, r
end

compressstore(v, ptr, m) = SIMD.Intrinsics.maskedcompressstore(v.data, ptr, m.data)

function bench(f, args...)
    f(args...)
    return minimum(@elapsed(f(args...)) for _ in 1:REPS) / KEYS * 1e9
end

native(f, types) = sprint(io -> code_native(io, f, types; debuginfo = :none))

function main()
    feats = [f for f in (:ssse3, :avx2, :avx512f, :avx512vl, :avx512vbmi2) if SIMD.Targets.has_feature(f)]
    println("Julia ", VERSION, ", LLVM ", Base.libllvm_version, ", CPU ", Sys.CPU_NAME,
            ", SIMD.Targets: ", isempty(feats) ? "portable only" : join(feats, " "))
    println()
    println("| lanes | scalar | llvm.masked.compressstore | vstorec | vstorec_overwrite | vpartition | vpermute → vperm/pshufb | vcompress → vpcompress |")
    println("|---|---:|---:|---:|---:|---:|:-:|:-:|")
    for (N, T) in ((8, UInt32), (4, UInt64), (8, Float32), (4, Float64), (16, UInt32),
                   (8, UInt64), (8, UInt16), (16, UInt8))
        x = T <: AbstractFloat ? rand(T, KEYS) : rand(T, KEYS)
        pivot = T <: AbstractFloat ? T(0.5) : typemax(T) >> 1
        lo = zeros(T, KEYS + 3N); hi = zeros(T, KEYS + 3N)
        ref = partition_scalar!(lo, hi, x, pivot)
        refset = sort(lo[1:ref]); hiset = sort(hi[1:(KEYS - ref)])
        ok(k) = k == ref && sort(lo[1:k]) == refset && sort(hi[1:(KEYS - k)]) == hiset
        t = Float64[]
        push!(t, bench(partition_scalar!, lo, hi, x, pivot))
        for store in (compressstore, vstorec, vstorec_overwrite)
            ok(partition_vstorec!(lo, hi, x, pivot, Val(N), store)) || error("wrong result")
            push!(t, bench(partition_vstorec!, lo, hi, x, pivot, Val(N), store))
        end
        k, r = partition_vpartition!(lo, hi, x, pivot, Val(N))
        (k == ref && sort(lo[1:k]) == refset && sort(hi[(r + 1):end]) == hiset) ||
            error("wrong result")
        push!(t, bench(partition_vpartition!, lo, hi, x, pivot, Val(N)))
        perm = occursin(r"vpermd|vpermps|pshufb", native(vpermute, Tuple{Vec{N,T},Vec{N,Int32}}))
        comp = occursin("vpcompress", native(vcompress, Tuple{Vec{N,T},Vec{N,Bool}}))
        @printf("| %d × %s | %s | %s | %s | %s | %s | %s | %s |\n", N, T,
                (@sprintf("%.2f", s) for s in t)..., perm ? "yes" : "no", comp ? "yes" : "no")
    end
    println()
    println("Times in ns per key for partitioning $(KEYS) keys (best of $REPS).")
end

main()
