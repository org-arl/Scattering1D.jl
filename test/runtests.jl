using Scattering1D, Test, Random, LinearAlgebra

include("oracle.jl")

relative_error(a, b) = norm(a - b) / norm(b)

function test_signal(n, fs, seed)
  rng = MersenneTwister(seed)
  t = (0:n-1) ./ fs
  tone = @. 3 * (1 + 0.8 * sin(2π * 37 * t)) * sin(2π * 3000 * t)
  chirp = @. 2 * sin(2π * (500 * t + 4000 * t^2))
  randn(rng, n) .+ tone .+ chirp
end

working_spec(bins; kw...) =
  ScatteringSpec(40_000, bins, Morlets(250, 20_000, 35), Morlets(4, Inf, 1; crossover=3.0); kw...)

@testset "Scattering1D" begin
  @testset "specification" begin
    spec = working_spec(225)
    @test length(frequencies(spec)) == 222
    @test frequencies(spec)[1] == 250
    @test bandwidths(spec) ≈ frequencies(spec) .* (2^(1 / 35) - 1)
    @test modulations(spec) == 4 .* 2 .^ (0:9)
    @test size(mask(spec)) == (10, 222)
    counts = [count(<=(6σ), modulations(spec)) for σ in bandwidths(spec)]
    @test vec(sum(mask(spec); dims=1)) == counts
    edge = (40_000 / 225 - 1) / 2
    @test times(spec, 40_000)[[1, end]] ≈ [edge, 40_000 - 1 - edge] ./ 40_000
    @test_throws ArgumentError ScatteringSpec(40_000, 225, Morlets(250, 30_000, 35), Morlets(4, Inf, 1))
    @test_throws ArgumentError ScatteringPlan(working_spec(127 * 3), zeros(Float32, 40_000, 1))
  end

  # With no downsampling of envelopes and no truncation of filters, the fast
  # transform and the slow one are the same computation.
  @testset "exact when the shortcuts are off" begin
    fs, n = 4096.0, 4096
    ψ1, ψ2 = Morlets(100, 2000, 8), Morlets(4, Inf, 1; crossover=3.0)
    spec = ScatteringSpec(fs, 64, ψ1, ψ2; reach=1e6, ε=1e-14)
    x = reshape(test_signal(n, fs, 1), :, 1)
    plan = ScatteringPlan(spec, x)
    c = scattering(plan, x)
    S0, S1, S2 = oracle(spec, x[:, 1], plan.geometry.np, plan.geometry.pad)
    @test relative_error(c.S0[:, 1], S0) < 1e-12
    @test relative_error(c.S1[:, :, 1], S1) < 1e-12
    @test relative_error(c.S2[:, :, :, 1], S2) < 1e-12
    tables = Scattering1D.Tables{Float64}(spec, n, Scattering1D.coarse_size)
    batched = scattering(Scattering1D.BatchedPlan(tables, x), x)
    @test relative_error(batched.S1[:, :, 1], S1) < 1e-12
    @test relative_error(batched.S2[:, :, :, 1], S2) < 1e-12
  end

  # The defaults trade a known amount of accuracy for speed. The slow
  # transform is run on every eighth band to keep the test short.
  @testset "default accuracy, $n samples" for (n, bins) in ((32_768, 256), (40_000, 225), (65_536, 256))
    spec = working_spec(bins)
    x = test_signal(n, spec.fs, 2)
    x32 = reshape(Float32.(x), :, 1)
    plan = ScatteringPlan(spec, x32)
    c = scattering(plan, x32)
    bands = 1:8:222
    S0, S1, S2 = oracle(spec, x, plan.geometry.np, plan.geometry.pad; bands)
    @test size(c.S2) == (bins, 10, 222, 1)
    @test relative_error(c.S0[:, 1], S0) < 1e-3
    @test relative_error(c.S1[:, bands, 1], S1[:, bands]) < 2e-3
    @test relative_error(c.S2[:, :, bands, 1], S2[:, :, bands]) < 1e-2
  end

  @testset "batches and first order" begin
    spec = working_spec(225)
    x = randn(MersenneTwister(3), Float32, 40_000, 5)
    plan = ScatteringPlan(spec, x)
    c = scattering(plan, x)
    for b in 1:5
      single = scattering(plan, x[:, b:b])
      @test single.S0 == c.S0[:, b:b]
      @test single.S1 == c.S1[:, :, b:b]
      @test single.S2 == c.S2[:, :, :, b:b]
    end
    f = first_order(plan, x)
    @test f.S0 == c.S0 && f.S1 == c.S1
    @test all(iszero, c.S2[:, .!mask(spec), :])
    @test all(!iszero, c.S2[:, mask(spec), :])
    fill!(c.S2, 7)
    scattering!(c, plan, x)
    @test all(==(7), c.S2[:, .!mask(spec), :])
    @test_throws DimensionMismatch scattering!(c, plan, x[:, 1:2])
    @test_throws DimensionMismatch scattering(plan, x[1:1000, :])
  end

  @testset "both engines agree" begin
    spec = working_spec(225)
    x = randn(MersenneTwister(4), Float32, 40_000, 2)
    tables = Scattering1D.Tables{Float32}(spec, 40_000, Scattering1D.coarse_size)
    a, b = scattering(Scattering1D.ThreadedPlan(tables), x), scattering(Scattering1D.BatchedPlan(tables, x), x)
    @test relative_error(b.S0, a.S0) < 1e-5
    @test relative_error(b.S1, a.S1) < 1e-5
    @test relative_error(b.S2, a.S2) < 1e-5
  end

  @testset "inference and allocation" begin
    spec = working_spec(225)
    for T in (Float32, Float64)
      x = randn(MersenneTwister(5), T, 40_000, 2)
      plan = ScatteringPlan(spec, x)
      c = @inferred scattering(plan, x)
      f = @inferred first_order(plan, x)
      @test c isa Coefficients{Matrix{T},Array{T,3},Array{T,4}}
      @test f isa FirstOrder{Matrix{T},Array{T,3}}
      @inferred scattering!(c, plan, x)
      @inferred first_order!(f, plan, x)
      # tasks are the only allocation, so it does not grow with the work done
      @test @allocated(scattering!(c, plan, x)) < 100_000
      batched = Scattering1D.BatchedPlan(Scattering1D.Tables{T}(spec, 40_000, Scattering1D.coarse_size), x)
      cb = @inferred scattering(batched, x)
      scattering!(cb, batched, x)
      @test @allocated(scattering!(cb, batched, x)) < 1000
    end
  end

  # The reverse pass applies the transpose of the transform's Jacobian, so its
  # result paired with any direction must equal the directional derivative.
  @testset "gradients" begin
    rng = MersenneTwister(6)
    gradients(c) = c isa Coefficients ? Coefficients(randn.(rng, size.((c.S0, c.S1, c.S2)))...) :
      FirstOrder(randn.(rng, size.((c.S0, c.S1)))...)
    paired(c̄, a, b, active) = dot(c̄.S0, a.S0 - b.S0) + dot(c̄.S1, a.S1 - b.S1) +
      (c̄ isa Coefficients ? dot(c̄.S2[:, active, :], (a.S2-b.S2)[:, active, :]) : 0)
    ψ1, ψ2 = Morlets(100, 2000, 8), Morlets(4, Inf, 1; crossover=3.0)
    cases = ((ScatteringSpec(4096, 64, ψ1, ψ2), 4096, 1), (ScatteringSpec(4096, 64, ψ1, ψ2), 4096, 5),
      (working_spec(225), 40_000, 2))
    for (spec, n, batch) in cases, transform in (scattering, first_order)
      x, δ, h = randn(rng, n, batch), randn(rng, n, batch), 1e-6
      plan = ScatteringPlan(spec, x)
      c̄ = gradients(transform(plan, x))
      x̄ = @inferred backpropagate(plan, x, c̄)
      derivative = paired(c̄, transform(plan, x + h * δ), transform(plan, x - h * δ), mask(spec)) / 2h
      @test isapprox(dot(x̄, δ), derivative; rtol=1e-6)
      tables = Scattering1D.Tables{Float64}(spec, n, Scattering1D.coarse_size)
      threaded, batched = Scattering1D.ThreadedPlan(tables), Scattering1D.BatchedPlan(tables, x)
      @test relative_error((@inferred backpropagate(batched, x, c̄)), backpropagate(threaded, x, c̄)) < 1e-10
    end
    spec = working_spec(225)
    x = randn(rng, Float32, 40_000, 2)
    plan = ScatteringPlan(spec, x)
    c̄ = Coefficients(randn.(rng, Float32, size.((scattering(plan, x).S0, scattering(plan, x).S1, scattering(plan, x).S2)))...)
    x̄ = backpropagate(plan, x, c̄)
    c̄.S2[:, .!mask(spec), :] .= 7
    # tasks add their shares in an order that varies, so rounding differs between runs
    @test backpropagate(plan, x, c̄) ≈ x̄
    @test_throws DimensionMismatch backpropagate(plan, x[:, 1:1], c̄)
    @test_throws DimensionMismatch backpropagate!(zeros(Float32, 40_000, 3), plan, x, c̄)
  end

  include("kymatio.jl")
end
