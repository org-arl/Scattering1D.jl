# Time the package on CPU threads. Run with the thread count to measure:
#   julia --project=benchmark -t 12 benchmark/cpu.jl

include("common.jl")

rows = benchmark("cpu", identity, (tables, x) -> Scattering1D.ThreadedPlan(tables))
save(rows, joinpath(@__DIR__, "results", "cpu_$(Threads.nthreads()).csv"))
foreach(println, rows)
