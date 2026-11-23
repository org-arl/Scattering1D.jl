# Compare sets of FFT lengths on the Apple GPU:
#   julia --project=devices/metal devices/metal/sweep.jl

using Metal

include(joinpath(@__DIR__, "..", "..", "benchmark", "common.jl"))
include(joinpath(@__DIR__, "..", "..", "benchmark", "sweep.jl"))

save(sweep(MtlArray; sync=() -> Metal.synchronize(), release=GC.gc), joinpath(@__DIR__, "results", "sweep.csv"))
