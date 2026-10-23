"""
    SIMD.Targets

Decides whether SIMD.jl may emit target-specific instructions (for example
AVX2's `vpermd`) where LLVM's portable lowering is slow. The answer is used
inside `@generated` functions, i.e. fixed when a method is compiled, so it
must describe the CPU that Julia generates code for:

- Only x86 features are queried (via `Base.BinaryPlatforms.CPUID`); on other
  architectures the portable lowering is always used.
- They are only used when Julia compiles for the host CPU (the default
  `--cpu-target=native`). Package images are compiled for, and validated
  against, the host CPU. Multi-target system images and `-C generic` get the
  portable lowering.
- Setting the environment variable `SIMD_JL_GENERIC=1` (before the methods
  are compiled) forces the portable lowering everywhere, e.g. for testing.

Every operation that uses a target-specific path also has a portable one
that gives identical results.
"""
module Targets

const CPUID = Base.BinaryPlatforms.CPUID
const IS_X86 = Sys.ARCH === :x86_64 || Sys.ARCH === :i686

"Whether `SIMD_JL_GENERIC` asks for the portable lowering."
force_generic() = lowercase(get(ENV, "SIMD_JL_GENERIC", "false")) in ("1", "true", "yes")

"Whether Julia generates code for the host CPU (`-C native`, the default)."
function native_codegen()
    p = Base.JLOptions().cpu_target
    p == C_NULL && return true
    t = unsafe_string(p)
    return isempty(t) || t == "native"
end

"""
    has_feature(f::Symbol) -> Bool

Whether target-specific code for x86 feature `f` (`:ssse3`, `:avx2`,
`:avx512f`, `:avx512vl`, ...) may be emitted; see [`SIMD.Targets`](@ref).
"""
function has_feature(f::Symbol)
    (IS_X86 && !force_generic() && native_codegen()) || return false
    flag = Symbol(:JL_X86_, f)
    isdefined(CPUID, flag) || return false
    return CPUID.test_cpu_feature(getfield(CPUID, flag))
end

end # module Targets
