# Lets Mooncake differentiate through `scatter` and `first_order` by using the
# ChainRulesCore rules, since it cannot differentiate the transform's own code.

module Scattering1DMooncakeExt

using Scattering1D, Mooncake, ChainRulesCore

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(scatter),ScatteringPlan,AbstractMatrix{<:AbstractFloat}}
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(first_order),ScatteringPlan,AbstractMatrix{<:AbstractFloat}}

end
