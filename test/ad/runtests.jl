# Gradients through the transform with each supported automatic-differentiation
# package, checked against finite differences. Run from the package folder:
#   julia --project=test/ad -e 'using Pkg; Pkg.instantiate()'
#   julia --project=test/ad test/ad/runtests.jl

using Scattering1D, Test, Random, LinearAlgebra
using ChainRulesCore, Zygote, Enzyme, Mooncake

const SPEC = ScatteringSpec(4096, 64, Morlets(100, 2000, 8), Morlets(4, Inf, 1; crossover=3.0))
const ACTIVE = mask(SPEC)

both_orders(plan, x) = (c = scattering(plan, x); sum(abs2, c.S0) + sum(log1p, c.S1) + sum(sqrt, c.S2[:, ACTIVE, :]))
first_only(plan, x) = (c = first_order(plan, x); sum(abs2, c.S1) + sum(c.S0))

function gradient_with(::Val{:Zygote}, loss, plan, x)
  Zygote.gradient(x -> loss(plan, x), x)[1]
end

function gradient_with(::Val{:Enzyme}, loss, plan, x)
  x̄ = zero(x)
  Enzyme.autodiff(Enzyme.Reverse, loss, Enzyme.Active, Enzyme.Const(plan), Enzyme.Duplicated(x, x̄))
  x̄
end

function gradient_with(::Val{:Mooncake}, loss, plan, x)
  Mooncake.value_and_gradient!!(Mooncake.build_rrule(loss, plan, x), loss, plan, x)[2][3]
end

@testset "automatic differentiation" begin
  rng = MersenneTwister(1)
  x, δ, h = randn(rng, 4096, 2), randn(rng, 4096, 2), 1e-6
  plan = ScatteringPlan(SPEC, x)
  @testset "$package, $loss" for package in (:Zygote, :Enzyme, :Mooncake), loss in (both_orders, first_only)
    derivative = (loss(plan, x + h * δ) - loss(plan, x - h * δ)) / 2h
    @test isapprox(dot(gradient_with(Val(package), loss, plan, x), δ), derivative; rtol=1e-6)
  end
  # a trainable step before the transform receives its gradient
  w = randn(rng, 4096)
  derivative = (both_orders(plan, x .* (w .+ h)) - both_orders(plan, x .* (w .- h))) / 2h
  @test isapprox(sum(Zygote.gradient(w -> both_orders(plan, x .* w), w)[1]), derivative; rtol=1e-6)
end
