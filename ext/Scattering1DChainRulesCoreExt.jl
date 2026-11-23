# Reverse-mode rules for automatic differentiation: gradients with respect to
# the signals pass through `scatter` and `first_order`. Plans are constants.

module Scattering1DChainRulesCoreExt

using Scattering1D, ChainRulesCore

ChainRulesCore.@non_differentiable ScatteringPlan(::Any...)
ChainRulesCore.@non_differentiable Coefficients(::ScatteringPlan, ::Any)
ChainRulesCore.@non_differentiable FirstOrder(::ScatteringPlan, ::Any)

function ChainRulesCore.rrule(::typeof(scatter), plan::ScatteringPlan, x::AbstractMatrix)
  c = scatter(plan, x)
  function scatter_pullback(c̄)
    c̄ = unthunk(c̄)
    filled = Coefficients(dense(c̄.S0, c.S0), dense(c̄.S1, c.S1), dense(c̄.S2, c.S2))
    NoTangent(), NoTangent(), backpropagate(plan, x, filled)
  end
  scatter_pullback(::AbstractZero) = (NoTangent(), NoTangent(), ZeroTangent())
  c, scatter_pullback
end

function ChainRulesCore.rrule(::typeof(first_order), plan::ScatteringPlan, x::AbstractMatrix)
  c = first_order(plan, x)
  function first_order_pullback(c̄)
    c̄ = unthunk(c̄)
    NoTangent(), NoTangent(), backpropagate(plan, x, FirstOrder(dense(c̄.S0, c.S0), dense(c̄.S1, c.S1)))
  end
  first_order_pullback(::AbstractZero) = (NoTangent(), NoTangent(), ZeroTangent())
  c, first_order_pullback
end

### helpers

# gradient of one set of coefficients as an array, zero where none was given
dense(ā::AbstractZero, a) = zero(a)
dense(ā, a) = convert(typeof(a), unthunk(ā))

end
