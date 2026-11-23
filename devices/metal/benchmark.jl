# Time the package on the Apple GPU:
#   julia --project=devices/metal devices/metal/benchmark.jl

using Metal

include(joinpath(@__DIR__, "..", "..", "benchmark", "common.jl"))

rows = benchmark("metal", MtlArray, Scattering1D.BatchedPlan; sync=() -> Metal.synchronize(), release=GC.gc)
save(rows, joinpath(@__DIR__, "results", "metal.csv"))
