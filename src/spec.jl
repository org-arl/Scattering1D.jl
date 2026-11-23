"""
    Wavelets(fmin, fmax, Q; crossover=1.0)

Constant-Q bank of Morlet wavelets with centre frequencies `fmin·2^(k/Q)` up
to `fmax` (Hz). Each wavelet has a Gaussian frequency response whose standard
deviation is chosen so that neighbouring responses cross `crossover` standard
deviations away from their centres.
"""
struct Wavelets
  fmin::Float64
  fmax::Float64
  Q::Int
  crossover::Float64
end

Wavelets(fmin, fmax, Q; crossover=1.0) = Wavelets(fmin, fmax, Q, crossover)

"""
    ScatteringSpec(fs, bins, ψ1, ψ2; cutoff=6.0, attenuation=0.1, reach=6.0,
      ε=1e-3, padding=2.5)

Second-order scattering transform of signals sampled at `fs` Hz, reported on
`bins` equally spaced time bins spanning the signal.

`ψ1` filters the signal into bands and `ψ2` filters each band's envelope. A
modulation wavelet is applied to a band only if its centre frequency is at
most `cutoff` times the band's bandwidth, and `ψ2.fmax` may be `Inf` to leave
that rule as the only limit. Every order is smoothed by a Gaussian low-pass
filter whose gain at the output Nyquist frequency is `attenuation`.

The remaining keywords trade accuracy for speed:
- `reach`: an envelope is sampled fast enough to hold its spectrum out to
  `reach` bandwidths of the filter that produced it.
- `ε`: filter responses below `ε` are treated as zero.
- `padding`: the signal is extended by reflection on each side by `padding`
  times the time-domain standard deviation of the longest filter.
"""
struct ScatteringSpec
  fs::Float64
  bins::Int
  ψ1::Wavelets
  ψ2::Wavelets
  cutoff::Float64
  attenuation::Float64
  reach::Float64
  ε::Float64
  padding::Float64
end

function ScatteringSpec(fs, bins, ψ1, ψ2; cutoff=6.0, attenuation=0.1, reach=6.0,
    ε=1e-3, padding=2.5
)
  0 < ψ1.fmin < ψ1.fmax <= fs / 2 || throw(ArgumentError("ψ1 must lie in (0, fs/2]"))
  0 < ψ2.fmin <= ψ2.fmax || throw(ArgumentError("ψ2 needs 0 < fmin ≤ fmax"))
  0 < attenuation < 1 && 0 < ε < 1 ||
    throw(ArgumentError("attenuation and ε must lie in (0, 1)"))
  bins > 0 && reach > 0 && padding >= 0 && cutoff > 0 ||
    throw(ArgumentError("bins, reach and cutoff must be positive, and padding not negative"))
  ScatteringSpec(fs, bins, ψ1, ψ2, cutoff, attenuation, reach, ε, padding)
end

"""
    frequencies(spec)

Centre frequencies (Hz) of the first-order bands, in the order of the band
dimension of the output.
"""
frequencies(spec::ScatteringSpec) = centres(spec.ψ1, spec.ψ1.fmax)

"""
    bandwidths(spec)

Standard deviations (Hz) of the first-order frequency responses.
"""
bandwidths(spec::ScatteringSpec) = frequencies(spec) .* relative_width(spec.ψ1)

"""
    modulations(spec)

Centre frequencies (Hz) of the modulation wavelets, in the order of the
modulation dimension of the output.
"""
function modulations(spec::ScatteringSpec)
  centres(spec.ψ2, min(spec.ψ2.fmax, spec.cutoff * maximum(bandwidths(spec))))
end

"""
    mask(spec)

Boolean matrix, modulation × band, that is `true` where the second-order
coefficient is computed. The other entries of the output are never written.
"""
function mask(spec::ScatteringSpec)
  counts = modulation_counts(spec)
  [k <= c for k in 1:maximum(counts; init=0), c in counts]
end

"""
    times(spec, n)

Centre times (s) of the output bins for signals of `n` samples.
"""
times(spec::ScatteringSpec, n) = (((0:spec.bins-1) .+ 0.5) .* (n / spec.bins) .- 0.5) ./ spec.fs

### helpers

relative_width(ψ::Wavelets) = (2^(1 / ψ.Q) - 1) / ψ.crossover

function centres(ψ::Wavelets, fmax)
  fmax < ψ.fmin && return Float64[]
  ψ.fmin .* exp2.((0:floor(Int, ψ.Q * log2(fmax / ψ.fmin))) ./ ψ.Q)
end

# number of modulation wavelets applied to each first-order band
function modulation_counts(spec::ScatteringSpec)
  K = length(modulations(spec))
  map(bandwidths(spec)) do σ
    spec.cutoff * σ < spec.ψ2.fmin && return 0
    min(K, floor(Int, spec.ψ2.Q * log2(spec.cutoff * σ / spec.ψ2.fmin)) + 1)
  end
end
