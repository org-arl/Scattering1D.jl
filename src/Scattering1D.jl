# Fast 1D scattering transform.
#
# One FFT of the reflect-padded signal is taken. Each wavelet then reads only
# the bins where its response matters and is inverted with a short FFT, which
# yields the band already downsampled. Envelopes are filtered again the same
# way for the second order, and every order is low-passed onto a common time
# axis.

module Scattering1D

using AbstractFFTs: plan_rfft, plan_brfft, plan_bfft!, plan_fft!
using LinearAlgebra: mul!
import FFTW

export Morlets, ScatteringSpec, ScatteringPlan, Coefficients, FirstOrder
export scattering, scattering!, first_order, first_order!, backpropagate, backpropagate!
export frequencies, bandwidths, modulations, mask, times

### types

"""
    ScatteringPlan(spec, x)

Precomputed filters, FFT plans and work buffers for applying `spec` to arrays
like `x`, a samples × batch matrix. The kind of array selects the device: an
`Array` is transformed on CPU threads, and any other array type with batched
transforms on its own device.

A plan for an `Array` accepts any batch size. A plan for another array type
accepts only the batch size of `x`. A plan must not be used from several
tasks at once.
"""
abstract type ScatteringPlan{T} end

"""
    Coefficients(plan, batch)

Zeroed output arrays for `scattering!`: `S0` (time × batch) is the low-passed
signal, `S1` (time × band × batch) the low-passed band envelopes, and `S2`
(time × modulation × band × batch) the low-passed modulation envelopes.
Entries of `S2` outside `mask(spec)` are never written.
"""
struct Coefficients{A0,A1,A2}
  S0::A0
  S1::A1
  S2::A2
end

"""
    FirstOrder(plan, batch)

Zeroed output arrays for `first_order!`: `S0` (time × batch) and `S1`
(time × band × batch).
"""
struct FirstOrder{A0,A1}
  S0::A0
  S1::A1
end

include("spec.jl")
include("tables.jl")
include("threaded.jl")
include("threaded_pullback.jl")
include("batched.jl")
include("batched_pullback.jl")

### public API

function Coefficients(plan::ScatteringPlan, batch)
  bins, slots, bands = dimensions(plan)
  Coefficients(zeros(plan, bins, batch), zeros(plan, bins, bands, batch),
    zeros(plan, bins, slots, bands, batch))
end

function FirstOrder(plan::ScatteringPlan, batch)
  bins, _, bands = dimensions(plan)
  FirstOrder(zeros(plan, bins, batch), zeros(plan, bins, bands, batch))
end

"""
    scattering(plan, x)

Scattering transform of each column of `x`, returned as `Coefficients`.
"""
scattering(plan::ScatteringPlan, x::AbstractMatrix) =
  scattering!(Coefficients(plan, size(x, 2)), plan, x)

"""
    scattering!(c, plan, x)

Scattering transform of each column of `x`, written into the `Coefficients`
`c`.
"""
function scattering! end

"""
    first_order(plan, x)

Zeroth and first order of the scattering transform of each column of `x`,
returned as `FirstOrder`. Skips all second-order work.
"""
first_order(plan::ScatteringPlan, x::AbstractMatrix) =
  first_order!(FirstOrder(plan, size(x, 2)), plan, x)

"""
    first_order!(c, plan, x)

Zeroth and first order of the scattering transform of each column of `x`,
written into the `FirstOrder` `c`.
"""
function first_order! end

"""
    backpropagate(plan, x, c̄)

Gradient with respect to the signals `x` of a scalar whose gradient with
respect to their scattering coefficients is `c̄`: the transpose of the
transform's Jacobian applied to `c̄`.

`c̄` is a `Coefficients` to go back through `scattering`, or a `FirstOrder` to go
back through `first_order`. Entries of `c̄.S2` outside `mask(spec)` are
ignored. The filters are constants: there is no gradient with respect to them.
"""
backpropagate(plan::ScatteringPlan, x::AbstractMatrix, c̄) = backpropagate!(similar(x), plan, x, c̄)

"""
    backpropagate!(x̄, plan, x, c̄)

As `backpropagate`, written into `x̄`.
"""
function backpropagate! end

end
