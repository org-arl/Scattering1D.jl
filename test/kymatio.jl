# Parity with kymatio: the package is given kymatio's own filters and
# settings, and its output is compared with coefficients written by
# test/kymatio/generate.py.

include("kymatio_tables.jl")

const KYMATIO = joinpath(@__DIR__, "kymatio")

@testset "kymatio parity, $n samples" for n in (32768, 40000, 65536)
  meta = kymatio_meta(joinpath(KYMATIO, "meta_$n.txt"))
  tables = kymatio_tables(meta)
  @test length(tables.bands) == meta[:bands]
  @test tables.slots == meta[:modulations]
  @test 1 + length(tables.bands) + length(tables.paths) == meta[:paths]
  reference = Array{Float32}(undef, tables.geometry.bins, meta[:paths])
  read!(joinpath(KYMATIO, "coefficients_$n.bin"), reference)
  x = reshape(kymatio_signal(n), :, 1)
  for plan in (Scattering1D.ThreadedPlan(tables), Scattering1D.BatchedPlan(tables, x))
    c = scattering(plan, x)
    S2 = reduce(hcat, (c.S2[:, p.slot, p.band, 1] for p in tables.paths))
    ours = hcat(c.S0[:, 1], c.S1[:, :, 1], S2)
    scale = maximum(abs, reference; dims=1)
    @test maximum(abs.(ours .- reference) ./ scale) < 1e-5
  end
end
