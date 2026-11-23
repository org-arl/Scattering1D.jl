# Time the batched engine with different sets of FFT lengths, to tune
# `Scattering1D.MANTISSAS` for a device. Included by the device scripts.

# ms per signal of the working transform for each set, at each batch size
function sweep(to_device; sync=() -> nothing, release=() -> nothing, batches=(16, 128))
  sets = ((8, 9, 10, 12, 15), (8, 10, 12, 14), (8, 10, 12), (8, 12), (8,))
  rows = ["mantissas,batch,groups,samples_per_signal,ms_per_signal"]
  for mantissas in sets, batch in batches
    x = to_device(randn(MersenneTwister(batch), Float32, N, batch))
    tables = Scattering1D.Tables{Float32}(WORKING, N, n -> Scattering1D.coarse_size(n, mantissas))
    plan = Scattering1D.BatchedPlan(tables, x)
    c = Coefficients(plan, batch)
    seconds = best_time(() -> (scatter!(c, plan, x); sync()))
    groups = length(plan.groups1) + length(plan.groups2)
    samples = sum(b.M for b in tables.bands) + sum(p.M for p in tables.paths)
    push!(rows, join((join(mantissas, "-"), batch, groups, samples, round(1e3 * seconds / batch; digits=4)), ","))
    println("  ", rows[end])
    flush(stdout)
    plan = c = nothing
    release()
  end
  rows
end
