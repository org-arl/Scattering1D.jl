# Lets Enzyme differentiate through `scattering` and `first_order` by using the
# ChainRulesCore rules, since it cannot differentiate the transform's own code.

module Scattering1DEnzymeExt

using Scattering1D, Enzyme, ChainRulesCore

Enzyme.@import_rrule(typeof(scattering), ScatteringPlan, AbstractMatrix)
Enzyme.@import_rrule(typeof(first_order), ScatteringPlan, AbstractMatrix)

end
