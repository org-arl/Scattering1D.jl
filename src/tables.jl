### types

# Time axis shared by every stage. Output sample j lies at padded-signal sample
# (row + j)·np/m + shift.
struct Geometry
  n::Int          # signal samples
  np::Int         # samples after padding
  pad::Int        # reflected samples placed before the signal
  m::Int          # output samples spanning the padded signal
  row::Int        # output sample of the first bin
  shift::Float64  # samples by which every output sample is delayed
  bins::Int       # output samples kept
end

# First-order filter, held only over the DFT bins where it matters.
struct Band
  k0::Int   # DFT bin of the padded signal at its first weight
  at::Int   # index of its first weight in `Tables.w1`
  len::Int  # number of weights
  M::Int    # samples of the band signal
end

# Modulation filter applied to one band's envelope spectrum.
struct Path
  band::Int
  slot::Int  # index along the modulation dimension
  k0::Int    # DFT bin of the envelope at its first weight
  at::Int    # index of its first weight in `Tables.w2`
  len::Int
  M::Int     # samples of the path signal
end

# Everything that defines a transform, independent of the device running it.
struct Tables{T}
  geometry::Geometry
  φ::Vector{Complex{T}}  # low-pass response from bin 0, including the output delay
  w1::Vector{T}          # band weights, scaled by 1/np
  bands::Vector{Band}
  w2::Vector{T}          # modulation weights
  paths::Vector{Path}    # sorted by band, then slot
  slots::Int             # length of the modulation dimension
end

### helpers

Tables{T}(t::Tables) where T = Tables{T}(t.geometry, t.φ, t.w1, t.bands, t.w2, t.paths, t.slots)

# Tables for `spec` applied to signals of `n` samples. `fftsize(k)` returns
# the FFT length to use for a signal that needs at least `k` samples.
Tables{T}(spec::ScatteringSpec, n, fftsize) where T =
  Tables{T}(spec, Geometry(spec, n), lowpass_width(spec, n), fftsize)

# As above on a given time axis, with a low-pass filter of bandwidth `σφ` (Hz).
function Tables{T}(spec::ScatteringSpec, geometry::Geometry, σφ, fftsize) where T
  f1, σ1, f2 = frequencies(spec), bandwidths(spec), modulations(spec)
  σ2 = f2 .* relative_width(spec.ψ2)
  counts = modulation_counts(spec)
  Δf = spec.fs / geometry.np
  z = sqrt(2log(1 / spec.ε))
  kφ = floor(Int, z * σφ / Δf)
  φ = [exp(-(k * Δf)^2 / 2σφ^2) * cispi(2k * geometry.shift / geometry.np) for k in 0:kφ]
  windows1 = [window(f, σ, Δf, spec.ε, (geometry.np - 1) ÷ 2) for (f, σ) in zip(f1, σ1)]
  windows2 = [window(f, σ, Δf, spec.ε, typemax(Int) ÷ 2) for (f, σ) in zip(f2, σ2)]
  ats2 = cumsum(vcat(1, [length(w) for (_, w) in windows2]))
  bands, paths, w1 = Band[], Path[], Float64[]
  for (i, (k0, w)) in enumerate(windows1)
    kenv = spec.reach * σ1[i] / Δf
    top = maximum(k0 + length(w) - 1 for (k0, w) in windows2[1:counts[i]]; init=0)
    M = min(fftsize(max(length(w), ceil(Int, kenv + min(kenv, max(kφ, top))))), geometry.np)
    push!(bands, Band(k0, length(w1) + 1, length(w), M))
    append!(w1, w ./ geometry.np)
    kclip = min(floor(Int, kenv), (M - 1) ÷ 2)
    for slot in 1:counts[i]
      k0, w = windows2[slot]
      lo, hi = max(k0, -kclip), min(k0 + length(w) - 1, kclip)
      len = max(hi - lo + 1, 0)
      kenv2 = spec.reach * σ2[slot] / Δf
      M2 = min(fftsize(max(len, ceil(Int, kenv2 + min(kenv2, kφ)), 2)), M)
      push!(paths, Path(i, slot, lo, ats2[slot] + lo - k0, len, M2))
    end
  end
  w2 = reduce(vcat, (w for (_, w) in windows2); init=Float64[])
  Tables{T}(geometry, φ, w1, bands, w2, paths, length(f2))
end

# bandwidth (Hz) of the low-pass filter: its gain at the output Nyquist
# frequency is `spec.attenuation`
lowpass_width(spec::ScatteringSpec, n) = spec.fs * spec.bins / 2n / sqrt(2log(1 / spec.attenuation))

# Padding long enough for the longest filter of `spec`, with a padded length
# that holds a whole number of output samples and suits the FFT.
function Geometry(spec::ScatteringSpec, n)
  σ1, σ2 = bandwidths(spec), modulations(spec) .* relative_width(spec.ψ2)
  used = maximum(modulation_counts(spec); init=0)
  σmin = minimum(vcat(σ1, σ2[1:used], lowpass_width(spec, n)))
  bins = spec.bins
  pad = min(ceil(Int, spec.padding * spec.fs / (2π * σmin)), n - 1)
  for m in cld(bins * (n + 2pad), n):fld(bins * (3n - 2), n)
    np, r = divrem(n * m, bins)
    r == 0 && iseven(np) && issmooth(np) && issmooth(m) || continue
    stride = np / m
    start = (np - n) ÷ 2 + (n / bins - 1) / 2
    row = floor(Int, start / stride)
    return Geometry(n, np, (np - n) ÷ 2, m, row, start - row * stride, bins)
  end
  throw(ArgumentError("no FFT-friendly padded length for $n samples and $bins bins"))
end

# Morlet response exp(-(f-fc)²/2σ²) - κ·exp(-f²/2σ²), zero at f = 0, sampled
# every `Δf` over the bins within ±klim where it exceeds `ε`. Returns the
# first bin and the samples.
function window(fc, σ, Δf, ε, klim)
  κ = exp(-fc^2 / 2σ^2)
  z = sqrt(2log(1 / ε))
  ks = max(ceil(Int, -z * σ / Δf), -klim):min(floor(Int, (fc + z * σ) / Δf), klim)
  w = [exp(-(k * Δf - fc)^2 / 2σ^2) - κ * exp(-(k * Δf)^2 / 2σ^2) for k in ks]
  keep = findfirst(>(ε) ∘ abs, w):findlast(>(ε) ∘ abs, w)
  ks[first(keep)], w[keep]
end

issmooth(n) = foldl((n, p) -> n ÷ p^(trailing_power(n, p)), (2, 3, 5); init=n) == 1

function trailing_power(n, p)
  k = 0
  while n % p == 0
    n ÷= p
    k += 1
  end
  k
end

# smallest even 5-smooth length that is at least `n`
function smooth_size(n)
  n = max(n + isodd(n), 2)
  while !issmooth(n)
    n += 2
  end
  n
end
