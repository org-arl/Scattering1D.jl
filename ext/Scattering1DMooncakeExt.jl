# Lets Mooncake differentiate through `scattering` and `first_order` by using the
# ChainRulesCore rules, since it cannot differentiate the transform's own code.

module Scattering1DMooncakeExt

using Scattering1D, Mooncake, ChainRulesCore

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(scattering),ScatteringPlan,AbstractMatrix{<:AbstractFloat}}
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(first_order),ScatteringPlan,AbstractMatrix{<:AbstractFloat}}

end
