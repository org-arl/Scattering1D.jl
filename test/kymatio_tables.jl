# Filter tables that make the package compute what kymatio computes: kymatio's
# own filters, pairing rule, padded length and sampling rates.
#
# The filter construction below follows kymatio/scattering1d/filter_bank.py
# (BSD 3-Clause, the Kymatio developers).

using FFTW, Statistics

const σ0, α, R_PSI = 0.1, 5.0, sqrt(0.5)

function kymatio_signal(n)
  noise, state = zeros(n), 12345
  for i in 1:n
    state = (1103515245 * state + 12345) % 2^31
    noise[i] = state / 2^31 - 0.5
  end
  t = (0:n-1) ./ n
  @. noise + sin(2π * (200 * t + 3000 * t * t)) * (1 + 0.5 * sin(2π * 13 * t))
end

# settings written by kymatio for one configuration, as `key = integer` lines
kymatio_meta(path) = Dict(Symbol(k) => parse(Int, v) for (k, v) in split.(readlines(path), " = "))

periodize(f, N) = vec(sum(reshape(f, N, :); dims=2))

function morlet_1d(N, ξ, σ)
  P = min(ceil(Int, sqrt(-2σ^2 * log(1e-7)) + 1), 5)
  freqs = ((1-P)*N:P*N-1) ./ N
  low = periodize(exp.(-(P == 1 ? fftfreq(N) : freqs) .^ 2 ./ 2σ^2), N) ./ (2P - 1)
  isnothing(ξ) && return low ./ sum(abs, ifft(low))
  gabor = periodize(exp.(-(freqs .- ξ) .^ 2 ./ 2σ^2), N) ./ (2P - 1)
  f = gabor .- gabor[1] / low[1] .* low
  f ./ sum(abs, ifft(f))
end

# center frequencies and bandwidths, in cycles per sample
function anden(J, Q)
  step = 2^(1 / Q)
  ξ = max(1 / (1 + 2^(3 / Q)), 0.35)
  σ = ξ * (1 - 1 / step) / (1 + 1 / step) / sqrt(2log(1 / R_PSI))
  σmin = σ0 / 2^J
  bank = Tuple{Float64,Float64}[]
  if σ <= σmin
    ξ = σ
  else
    push!(bank, (ξ, σ))
    while σ > σmin * step
      ξ /= step
      σ /= step
      push!(bank, (ξ, σ))
    end
  end
  elbow = ξ
  for _ in 1:Q-1
    ξ -= elbow / Q
    push!(bank, (ξ, σmin))
  end
  bank
end

dyadic_subsampling(ξ, σ) = floor(Int, -log2(min(abs(ξ) + α * σ, 0.5))) - 1

# First bin and weights of the periodic response `f` over the bins where it
# exceeds `threshold`, shortened from its weaker end to at most `limit` bins.
function circular_window(f, limit, threshold)
  N = length(f)
  kept = findall(>(threshold) ∘ abs, f) .- 1
  gap, i = findmax(diff(vcat(kept, kept[1] + N)))
  k0, len = kept[mod1(i + 1, length(kept))], N - gap + 1
  w = [f[mod(k0 + d, N) + 1] for d in 0:len-1]
  while length(w) > limit
    if abs(w[1]) < abs(w[end])
      popfirst!(w)
      k0 += 1
    else
      pop!(w)
    end
  end
  k0, w
end

function kymatio_tables(meta)
  n, np, stride = meta[:n], meta[:padded], meta[:log2_stride]
  T = 2^meta[:J]
  bins = meta[:last] - meta[:first]
  geometry = Scattering1D.Geometry(n, np, meta[:pad_left], np >> stride, meta[:first], 0.0, bins)
  φ = morlet_1d(np, nothing, σ0 / T)
  φ = complex.(φ[1:findlast(>(1e-13), φ[1:np÷2])])
  bank1, bank2 = anden(meta[:J], meta[:Q1]), anden(meta[:J], meta[:Q2])
  j1s, j2s = [dyadic_subsampling(b...) for b in bank1], [dyadic_subsampling(b...) for b in bank2]
  ψ2 = [morlet_1d(np, b...) for b in bank2]
  bands, paths, w1, w2 = Scattering1D.Band[], Scattering1D.Path[], Float64[], Float64[]
  for (i, (ξ, σ)) in enumerate(bank1)
    k1 = min(j1s[i], stride)
    M = np >> k1
    k0, w = circular_window(morlet_1d(np, ξ, σ), M, 1e-13)
    push!(bands, Scattering1D.Band(k0, length(w1) + 1, length(w), M))
    append!(w1, w ./ np)
    for slot in findall(>(j1s[i]), j2s)
      M2 = M >> max(min(j2s[slot], stride) - k1, 0)
      k0, w = circular_window(periodize(ψ2[slot], M), M2, 1e-13)
      push!(paths, Scattering1D.Path(i, slot, k0, length(w2) + 1, length(w), M2))
      append!(w2, w)
    end
  end
  Scattering1D.Tables{Float64}(geometry, φ, w1, bands, w2, paths, length(bank2))
end
