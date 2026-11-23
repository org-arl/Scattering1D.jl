# Timing shared by the CPU, Metal and CUDA benchmarks.
#
# Three transforms of 40,000-sample signals are timed over batch sizes 1 to
# 128: the working parameters, and the two kymatio configurations of
# kymatio_bench.py computed with kymatio's own filters, so that those timings
# compare like with like.

using Scattering1D, Random, Printf

include(joinpath(@__DIR__, "..", "test", "kymatio_tables.jl"))

const N = 40_000
const BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
const WORKING = ScatteringSpec(40_000, 225, Wavelets(250, 20_000, 35), Wavelets(4, Inf, 1; crossover=3.0))

# shortest of repeated runs, in seconds, stopping after `budget` seconds
function best_time(run; budget=5.0)
  run()
  best, spent, runs = Inf, 0.0, 0
  while runs < 3 || (spent < budget && runs < 50)
    elapsed = @elapsed run()
    best, spent, runs = min(best, elapsed), spent + elapsed, runs + 1
  end
  best
end

# Time every transform on arrays made by `to_device`. `sync` waits for the
# device to finish, `release` frees device memory between plans, and
# `plan_tables` builds a plan from filter tables for that device.
function benchmark(label, to_device, plan_tables; sync=() -> nothing, release=() -> nothing, batches=BATCHES)
  rows = ["config,device,threads,batch,order,ms_per_signal"]
  record(config, batch, order, seconds) = begin
    ms = 1e3 * seconds / batch
    push!(rows, @sprintf("%s,%s,%d,%d,%s,%.4f", config, label, Threads.nthreads(), batch, order, ms))
    @printf("  %-8s batch %3d %-6s %8.3f ms/signal\n", config, batch, order, ms)
    flush(stdout)
  end
  for batch in batches
    x = to_device(randn(MersenneTwister(batch), Float32, N, batch))
    plan = ScatteringPlan(WORKING, x)
    c, f = Coefficients(plan, batch), FirstOrder(plan, batch)
    record("working", batch, "both", best_time(() -> (scatter!(c, plan, x); sync())))
    record("working", batch, "first", best_time(() -> (first_order!(f, plan, x); sync())))
    plan = c = f = nothing
    release()
    for tag in ("kymatio", "working")
      meta = joinpath(@__DIR__, "kymatio", "meta_$tag.txt")
      tables = kymatio_tables(kymatio_meta(meta))
      plan = plan_tables(Scattering1D.Tables{Float32}(tables), x)
      c = Coefficients(plan, batch)
      record("as-kymatio-$tag", batch, "both", best_time(() -> (scatter!(c, plan, x); sync())))
      plan = c = nothing
      release()
    end
  end
  rows
end

function save(rows, path)
  mkpath(dirname(path))
  write(path, join(rows, "\n") * "\n")
end
