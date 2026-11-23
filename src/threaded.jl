### types

# Run of consecutive bins copied from a half spectrum into a filter window.
# With step -1 the run is read backwards and conjugated, which is how bins of
# the half that is not stored are obtained.
struct Segment
  dst::Int32
  src::Int32
  len::Int32
  step::Int32
end

# What one filter needs at run time.
struct Stage
  at::Int                     # index of its first weight
  len::Int                    # number of weights
  size::Int                   # index of its FFT length in `ThreadedPlan.sizes`
  segments::UnitRange{Int32}  # its runs in the segment list
  kmax::Int                   # last envelope bin passed to the low-pass filter
  slot::Int                   # position of its output along the modulation dimension
end

# Buffers owned by one task.
struct Work{T}
  x::Vector{T}                     # padded signal
  X::Vector{Complex{T}}            # its spectrum
  z::Vector{Vector{Complex{T}}}    # band or path signal, one per FFT length
  e::Vector{Vector{T}}             # envelope, one per FFT length
  E1::Vector{Vector{Complex{T}}}   # band envelope spectrum, one per FFT length
  E2::Vector{Vector{Complex{T}}}   # path envelope spectrum, one per FFT length
  F::Vector{Complex{T}}            # spectrum of the output samples
  r::Vector{T}                     # output samples spanning the padded signal
end

# Buffers of the reverse pass owned by one task.
struct ReverseWork{T}
  z::Vector{Vector{Complex{T}}}  # band signal, one per FFT length
  g::Vector{Vector{Complex{T}}}  # gradient of a band's envelope spectrum, one per FFT length
  X̄::Vector{Complex{T}}          # gradient of the padded signal's spectrum
end

# FFT plans and buffers that only the reverse pass needs.
struct Reverse{T,PS,PE,PO,PX}
  spread::Vector{PS}    # gradient of a band or path signal to that of its window, per FFT length
  envelope::Vector{PE}  # gradient of an envelope spectrum to that of the envelope, per FFT length
  output::PO            # gradient of the output samples to that of their spectrum
  signal::PX            # gradient of the padded signal's spectrum to that of the padded signal
  works::Vector{Tuple{Work{T},ReverseWork{T}}}
end

# Plan for `Array` input: every band of every signal is a unit of work, and
# units are shared out among tasks.
struct ThreadedPlan{T,PX,PZ,PE,PF,R<:Reverse} <: ScatteringPlan{T}
  geometry::Geometry
  slots::Int
  φ::Vector{Complex{T}}
  w1::Vector{T}
  w2::Vector{T}
  bands::Vector{Stage}
  paths::Vector{Stage}
  band_paths::Vector{UnitRange{Int}}  # paths of each band
  segments1::Vector{Segment}
  segments2::Vector{Segment}
  sizes::Vector{Int}
  forward::PX                 # padded signal to spectrum
  inverse::Vector{PZ}         # window to band or path signal, one per FFT length
  envelope::Vector{PE}        # envelope to spectrum, one per FFT length
  output::PF                  # spectrum to output samples
  X::Vector{Complex{T}}       # spectra of all signals of the batch, one after another
  works::Vector{Work{T}}
  reverse::Base.RefValue{Union{Nothing,R}}  # built at the first reverse pass
end

### public API

function ScatteringPlan(spec::ScatteringSpec, x::Matrix{T}) where T<:AbstractFloat
  ThreadedPlan(Tables{T}(spec, size(x, 1), smooth_size))
end

function scatter!(c::Coefficients, plan::ThreadedPlan, x::AbstractMatrix)
  check_sizes(plan, x, c.S0, c.S1)
  size(c.S2) == (plan.geometry.bins, plan.slots, length(plan.bands), size(x, 2)) ||
    throw(DimensionMismatch("S2 does not match the plan and batch"))
  transform!(plan, x, c.S0, c.S1, c.S2)
  c
end

function first_order!(c::FirstOrder, plan::ThreadedPlan, x::AbstractMatrix)
  check_sizes(plan, x, c.S0, c.S1)
  transform!(plan, x, c.S0, c.S1, nothing)
  c
end

### helpers

function ThreadedPlan(t::Tables{T}) where T
  g = t.geometry
  sizes = sort!(unique!(vcat([b.M for b in t.bands], [p.M for p in t.paths])))
  index = Dict(M => i for (i, M) in enumerate(sizes))
  kφ = length(t.φ) - 1
  segments1, segments2 = Segment[], Segment[]
  bands = map(t.bands) do b
    runs = runs!(segments1, b.k0, b.len, g.np)
    Stage(b.at, b.len, index[b.M], runs, min(kφ, (b.M - 1) ÷ 2), 0)
  end
  paths = map(t.paths) do p
    runs = runs!(segments2, p.k0, p.len, t.bands[p.band].M)
    Stage(p.at, p.len, index[p.M], runs, min(kφ, (p.M - 1) ÷ 2), p.slot)
  end
  band_paths = [searchsorted([p.band for p in t.paths], i) for i in eachindex(t.bands)]
  threads = FFTW.get_num_threads()
  FFTW.set_num_threads(1)
  forward = plan_rfft(Vector{T}(undef, g.np); flags=FFTW.MEASURE)
  inverse = [plan_bfft!(Vector{Complex{T}}(undef, M); flags=FFTW.MEASURE) for M in sizes]
  envelope = [plan_rfft(Vector{T}(undef, M); flags=FFTW.MEASURE) for M in sizes]
  output = plan_brfft(Vector{Complex{T}}(undef, g.m ÷ 2 + 1), g.m; flags=FFTW.MEASURE)
  FFTW.set_num_threads(threads)
  works = [Work{T}(sizes, g) for _ in 1:Threads.nthreads()]
  R = Reverse{T,typeof(plan_fft!(zeros(Complex{T}, 2))),typeof(output),typeof(forward),typeof(output)}
  ThreadedPlan(g, t.slots, t.φ, t.w1, t.w2, bands, paths, band_paths, segments1, segments2,
    sizes, forward, inverse, envelope, output, Complex{T}[], works, Ref{Union{Nothing,R}}(nothing))
end

dimensions(plan::ThreadedPlan) = (plan.geometry.bins, plan.slots, length(plan.bands))
Base.zeros(::ThreadedPlan{T}, dims...) where T = zeros(T, dims...)

function Work{T}(sizes, g::Geometry) where T
  spectra() = [Vector{Complex{T}}(undef, M ÷ 2 + 1) for M in sizes]
  Work{T}(Vector{T}(undef, g.np), Vector{Complex{T}}(undef, g.np ÷ 2 + 1),
    [Vector{Complex{T}}(undef, M) for M in sizes], [Vector{T}(undef, M) for M in sizes],
    spectra(), spectra(), Vector{Complex{T}}(undef, g.m ÷ 2 + 1), Vector{T}(undef, g.m))
end

# Append to `segments` the runs that read bins k0, k0+1, … of a real signal's
# spectrum of period `period` from its stored half, and return their range.
function runs!(segments, k0, len, period)
  start = length(segments) + 1
  for d in 0:len-1
    bin = mod(k0 + d, period)
    src, step = bin <= period ÷ 2 ? (bin + 1, 1) : (period - bin + 1, -1)
    prev = length(segments) < start ? nothing : segments[end]
    if !isnothing(prev) && prev.step == step && prev.src + step * prev.len == src
      segments[end] = Segment(prev.dst, prev.src, prev.len + 1, step)
    else
      push!(segments, Segment(d + 1, src, 1, step))
    end
  end
  Int32(start):Int32(length(segments))
end

function check_sizes(plan, x, S0, S1)
  g, batch = plan.geometry, size(x, 2)
  size(x, 1) == g.n || throw(DimensionMismatch("plan is for signals of $(g.n) samples"))
  size(S0) == (g.bins, batch) && size(S1) == (g.bins, length(plan.bands), batch) ||
    throw(DimensionMismatch("coefficients do not match the plan and batch"))
end

function transform!(plan::ThreadedPlan, x, S0, S1, S2)
  g, batch, nbands = plan.geometry, size(x, 2), length(plan.bands)
  nbins = g.np ÷ 2 + 1
  resize!(plan.X, nbins * batch)
  share(plan.works, batch) do work, b
    spectrum!(plan, work, x, b)
    copyto!(plan.X, (b - 1) * nbins + 1, work.X, 1, nbins)
    emit!(S0, (b - 1) * g.bins, plan, work, work.X, length(plan.φ) - 1, inv(g.np))
  end
  share(plan.works, batch * nbands) do work, unit
    b, i = (unit - 1) % batch + 1, nbands - (unit - 1) ÷ batch
    X = view(plan.X, (b - 1) * nbins + 1:b * nbins)
    band!(S1, S2, plan, work, X, i, b)
  end
end

# Run f(work, unit) for every unit, handing units to as many tasks as there
# are work buffers.
function share(f::F, works, units) where F
  if min(length(works), units) <= 1
    foreach(unit -> f(works[1], unit), 1:units)
    return
  end
  next = Threads.Atomic{Int}(1)
  @sync for work in view(works, 1:min(length(works), units))
    Threads.@spawn while true
      unit = Threads.atomic_add!(next, 1)
      unit > units && break
      f(work, unit)
    end
  end
end

# reflect-pad signal b and transform it
function spectrum!(plan, work, x, b)
  g, xp = plan.geometry, work.x
  @inbounds for t in 1:g.pad
    xp[t] = x[g.pad - t + 2, b]
  end
  @inbounds for t in 1:g.n
    xp[g.pad + t] = x[t, b]
  end
  @inbounds for t in 1:g.np-g.pad-g.n
    xp[g.pad + g.n + t] = x[g.n - t, b]
  end
  mul!(work.X, plan.forward, xp)
end

function band!(S1, S2, plan, work, X, i, b)
  bins, nbands = plan.geometry.bins, length(plan.bands)
  band = plan.bands[i]
  M = plan.sizes[band.size]
  E = work.E1[band.size]
  envelope!(E, plan, work, X, plan.w1, band, plan.segments1, inv(M))
  emit!(S1, bins * (i - 1 + nbands * (b - 1)), plan, work, E, band.kmax, 1)
  isnothing(S2) && return
  for p in plan.band_paths[i]
    path = plan.paths[p]
    E2 = work.E2[path.size]
    envelope!(E2, plan, work, E, plan.w2, path, plan.segments2, 1)
    at = bins * (path.slot - 1 + plan.slots * (i - 1 + nbands * (b - 1)))
    emit!(S2, at, plan, work, E2, path.kmax, inv(plan.sizes[path.size]))
  end
end

# Filter the spectrum `X` with `stage`, take the magnitude of the result times
# `scale`, and write that envelope's spectrum to `E`.
function envelope!(E, plan, work, X, w, stage, segments, scale)
  z, e = work.z[stage.size], work.e[stage.size]
  filter!(z, plan, X, w, stage, segments)
  c = convert(eltype(e), scale)
  @inbounds @simd for j in eachindex(e)
    e[j] = sqrt(abs2(z[j])) * c
  end
  mul!(E, plan.envelope[stage.size], e)
end

# Filter the spectrum `X` with `stage` and write the resulting signal to `z`.
function filter!(z, plan, X, w, stage, segments)
  @inbounds for s in stage.segments
    seg = segments[s]
    dst, src, wat = Int(seg.dst), Int(seg.src), stage.at + seg.dst - 1
    if seg.step > 0
      @simd for j in 0:seg.len-1
        z[dst + j] = X[src + j] * w[wat + j]
      end
    else
      @simd for j in 0:seg.len-1
        z[dst + j] = conj(X[src - j]) * w[wat + j]
      end
    end
  end
  @inbounds z[stage.len+1:end] .= 0
  plan.inverse[stage.size] * z
end

# Low-pass the envelope spectrum `E` (bins 0 to kmax), scale it, and write the
# output samples to `S` after linear index `at`.
function emit!(S, at, plan, work, E, kmax, scale)
  g, φ, F = plan.geometry, plan.φ, work.F
  m, half = g.m, g.m ÷ 2
  c = convert(real(eltype(F)), scale)
  fill!(F, 0)
  @inbounds @simd for k in 0:min(kmax, half)
    F[k + 1] = E[k + 1] * φ[k + 1] * c
  end
  # bins at and beyond the output Nyquist frequency alias onto lower ones
  @inbounds for k in half:kmax
    v = E[k + 1] * φ[k + 1] * c
    plain, mirrored = mod(k, m), mod(-k, m)
    k > half && plain <= half && (F[plain + 1] += v)
    mirrored <= half && (F[mirrored + 1] += conj(v))
  end
  mul!(work.r, plan.output, F)
  copyto!(S, at + 1, work.r, g.row + 1, g.bins)
end
