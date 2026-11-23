# Reverse pass of the threaded engine: every step of the transform is undone
# by its transpose, in the opposite order.

### public API

function backpropagate!(x̄::AbstractMatrix, plan::ThreadedPlan{T}, x::AbstractMatrix, c̄) where T
  check_sizes(plan, x, c̄.S0, c̄.S1)
  size(x̄) == size(x) || throw(DimensionMismatch("gradient and signals differ in size"))
  S̄2 = second_order(c̄)
  isnothing(S̄2) || size(S̄2) == (plan.geometry.bins, plan.slots, length(plan.bands), size(x, 2)) ||
    throw(DimensionMismatch("S2 does not match the plan and batch"))
  reverse = Reverse(plan)
  batch, nbands = size(x, 2), length(plan.bands)
  if batch >= length(reverse.works) || length(reverse.works) == 1
    share(reverse.works, batch) do (work, rwork), b
      spectrum!(plan, work, x, b)
      fill!(rwork.X̄, 0)
      foreach(i -> band_pullback!(rwork.X̄, c̄.S1, S̄2, plan, reverse, work, rwork, work.X, i, b), 1:nbands)
      signal_pullback!(x̄, c̄.S0, plan, reverse, work, rwork.X̄, b)
    end
  else
    # too few signals to give each task its own: share the bands of each signal
    work, rwork = reverse.works[1]
    for b in 1:batch
      X = spectrum!(plan, work, x, b)
      foreach(w -> fill!(w[2].X̄, 0), reverse.works)
      share(reverse.works, nbands) do (work, rwork), i
        band_pullback!(rwork.X̄, c̄.S1, S̄2, plan, reverse, work, rwork, X, i, b)
      end
      foreach(w -> rwork.X̄ .+= w[2].X̄, view(reverse.works, 2:length(reverse.works)))
      signal_pullback!(x̄, c̄.S0, plan, reverse, work, rwork.X̄, b)
    end
  end
  x̄
end

### helpers

second_order(c̄::Coefficients) = c̄.S2
second_order(::FirstOrder) = nothing

# The reverse parts of `plan`, built at the first call.
function Reverse(plan::ThreadedPlan{T}) where T
  reverse = plan.reverse[]
  isnothing(reverse) || return reverse
  g = plan.geometry
  threads = FFTW.get_num_threads()
  FFTW.set_num_threads(1)
  flags = FFTW.MEASURE
  spread = [plan_fft!(Vector{Complex{T}}(undef, M); flags) for M in plan.sizes]
  envelope = [plan_brfft(Vector{Complex{T}}(undef, M ÷ 2 + 1), M; flags) for M in plan.sizes]
  output = plan_rfft(Vector{T}(undef, g.m); flags)
  signal = plan_brfft(Vector{Complex{T}}(undef, g.np ÷ 2 + 1), g.np; flags)
  FFTW.set_num_threads(threads)
  works = map(plan.works) do work
    rwork = ReverseWork{T}([Vector{Complex{T}}(undef, M) for M in plan.sizes],
      [Vector{Complex{T}}(undef, M ÷ 2 + 1) for M in plan.sizes], Vector{Complex{T}}(undef, g.np ÷ 2 + 1))
    (work, rwork)
  end
  plan.reverse[] = Reverse(spread, envelope, output, signal, works)
end

# Add to `X̄` the gradient of signal b's spectrum that comes through band i.
function band_pullback!(X̄, S̄1, S̄2, plan, reverse, work, rwork, X, i, b)
  bins, nbands = plan.geometry.bins, length(plan.bands)
  band = plan.bands[i]
  M = plan.sizes[band.size]
  z, e, E, Ē = rwork.z[band.size], work.e[band.size], work.E1[band.size], rwork.g[band.size]
  filter!(z, plan, X, plan.w1, band, plan.segments1)
  fill!(Ē, 0)
  if !isnothing(S̄2)
    c = convert(eltype(e), inv(M))
    @inbounds @simd for j in eachindex(e)
      e[j] = sqrt(abs2(z[j])) * c
    end
    mul!(E, plan.envelope[band.size], e)
    for p in plan.band_paths[i]
      path = plan.paths[p]
      u, Ē2 = work.z[path.size], work.E2[path.size]
      filter!(u, plan, E, plan.w2, path, plan.segments2)
      fill!(Ē2, 0)
      at = bins * (path.slot - 1 + plan.slots * (i - 1 + nbands * (b - 1)))
      absorb!(Ē2, S̄2, at, plan, reverse, work, path.kmax, inv(plan.sizes[path.size]))
      spread!(Ē, Ē2, u, plan, reverse, work, plan.w2, path, plan.segments2, 1)
    end
  end
  absorb!(Ē, S̄1, bins * (i - 1 + nbands * (b - 1)), plan, reverse, work, band.kmax, 1)
  spread!(X̄, Ē, z, plan, reverse, work, plan.w1, band, plan.segments1, inv(M))
end

# Transpose of `emit!`: add to `Ē` (bins 0 to kmax) the gradient that the
# output samples of `S̄` after linear index `at` pass to an envelope spectrum.
function absorb!(Ē, S̄, at, plan, reverse, work, kmax, scale)
  g, φ, F, r = plan.geometry, plan.φ, work.F, work.r
  m, half = g.m, g.m ÷ 2
  c = convert(real(eltype(F)), scale)
  fill!(r, 0)
  copyto!(r, g.row + 1, S̄, at + 1, g.bins)
  mul!(F, reverse.output, r)
  # a bin strictly between 0 and the Nyquist frequency stands for two
  @inbounds for k in 2:(iseven(m) ? half : half + 1)
    F[k] *= 2
  end
  @inbounds @simd for k in 0:min(kmax, half)
    Ē[k + 1] += conj(φ[k + 1]) * F[k + 1] * c
  end
  @inbounds for k in half:kmax
    plain, mirrored = mod(k, m), mod(-k, m)
    k > half && plain <= half && (Ē[k + 1] += conj(φ[k + 1]) * F[plain + 1] * c)
    mirrored <= half && (Ē[k + 1] += conj(φ[k + 1] * F[mirrored + 1]) * c)
  end
end

# Transpose of `envelope!`: given the gradient `Ē` of an envelope spectrum and
# the signal `z` whose magnitude times `scale` is that envelope, add to `X̄`
# the gradient of the spectrum that `stage` filtered to make `z`. Overwrites
# `Ē` and `z`.
function spread!(X̄, Ē, z, plan, reverse, work, w, stage, segments, scale)
  e = work.e[stage.size]
  M = length(z)
  @inbounds for k in 2:(iseven(M) ? M ÷ 2 : M ÷ 2 + 1)
    Ē[k] /= 2
  end
  mul!(e, reverse.envelope[stage.size], Ē)
  c = convert(eltype(e), scale)
  @inbounds @simd for j in eachindex(z)
    a = sqrt(abs2(z[j]))
    z[j] *= ifelse(a > 0, e[j] * c / a, zero(a))
  end
  reverse.spread[stage.size] * z
  @inbounds for s in stage.segments
    seg = segments[s]
    dst, src, wat = Int(seg.dst), Int(seg.src), stage.at + seg.dst - 1
    if seg.step > 0
      @simd for j in 0:seg.len-1
        X̄[src + j] += z[dst + j] * w[wat + j]
      end
    else
      @simd for j in 0:seg.len-1
        X̄[src - j] += conj(z[dst + j]) * w[wat + j]
      end
    end
  end
end

# Finish signal b: add the gradient through the low-passed signal to `X̄`,
# and take the result back through the transform and the padding.
function signal_pullback!(x̄, S̄0, plan, reverse, work, X̄, b)
  g, xp = plan.geometry, work.x
  absorb!(X̄, S̄0, (b - 1) * g.bins, plan, reverse, work, length(plan.φ) - 1, inv(g.np))
  @inbounds for k in 2:g.np÷2
    X̄[k] /= 2
  end
  mul!(xp, reverse.signal, X̄)
  @inbounds for t in 1:g.n
    x̄[t, b] = xp[g.pad + t]
  end
  @inbounds for t in 1:g.pad
    x̄[g.pad - t + 2, b] += xp[t]
  end
  @inbounds for t in 1:g.np-g.pad-g.n
    x̄[g.n - t, b] += xp[g.pad + g.n + t]
  end
end
