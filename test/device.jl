# Checks for a GPU array type, run from an environment that has the device
# package: the batched engine on the device must agree with the threaded
# engine on the CPU when both use the same filter tables.

using Scattering1D, Test, Random, LinearAlgebra

function device_tests(to_device)
  spec = ScatteringSpec(40_000, 225, Wavelets(250, 20_000, 35), Wavelets(4, Inf, 1; crossover=3.0))
  @testset "batch of $batch" for batch in (1, 3)
    x = randn(MersenneTwister(batch), Float32, 40_000, batch)
    xd = to_device(x)
    tables = Scattering1D.Tables{Float32}(spec, size(x, 1), Scattering1D.coarse_size)
    expected = scatter(Scattering1D.ThreadedPlan(tables), x)
    plan = ScatteringPlan(spec, xd)
    @test plan isa Scattering1D.BatchedPlan
    c = @inferred scatter(plan, xd)
    @test typeof(c.S2) == typeof(to_device(expected.S2))
    for name in (:S0, :S1, :S2)
      ours, theirs = Array(getfield(c, name)), getfield(expected, name)
      @test norm(ours - theirs) < 1e-5 * norm(theirs)
    end
    f = @inferred first_order(plan, xd)
    # GPU transforms need not repeat bit for bit
    @test Array(f.S1) ≈ Array(c.S1)
    @test_throws DimensionMismatch scatter(plan, to_device(randn(Float32, 40_000, batch + 1)))
  end
end
