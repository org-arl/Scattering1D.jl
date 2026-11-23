# Slow evaluation of the transform that shares no machinery with the package:
# every filter is applied over the whole spectrum with full-length FFTs, no
# signal is downsampled before the output is read, and no filter is truncated.

using FFTW

morlet(f, fc, σ) = exp(-(f - fc)^2 / 2σ^2) - exp(-fc^2 / 2σ^2) * exp(-f^2 / 2σ^2)

function reflect_pad(x, before, after)
  n = length(x)
  vcat(x[before+1:-1:2], x, x[n-1:-1:n-after])
end

# values at the fractional sample positions `ts` of the signal with spectrum
# `Y`, using one inverse FFT per distinct fractional offset
function sample(Y, ts)
  np = length(Y)
  k = fftfreq(np, np)
  out = zeros(length(ts))
  δs = round.(ts .- floor.(ts); digits=9)
  for δ in unique(δs)
    y = real(ifft(Y .* cispi.(2 .* k .* δ ./ np)))
    for j in findall(==(δ), δs)
      out[j] = y[floor(Int, ts[j]) + 1]
    end
  end
  out
end

# Coefficients of `x` padded to `np` samples with `pad` before the signal.
# Only the first-order bands listed in `bands` are computed.
function oracle(spec, x::AbstractVector, np, pad; bands=eachindex(frequencies(spec)))
  n = length(x)
  f = fftfreq(np, spec.fs)
  f1, σ1, f2 = frequencies(spec), bandwidths(spec), modulations(spec)
  σ2 = f2 .* (2^(1 / spec.ψ2.Q) - 1) ./ spec.ψ2.crossover
  active = mask(spec)
  σφ = spec.fs * spec.bins / 2n / sqrt(2log(1 / spec.attenuation))
  φ = @. exp(-f^2 / 2σφ^2)
  ts = pad .+ ((0:spec.bins-1) .+ 0.5) .* (n / spec.bins) .- 0.5
  X = fft(reflect_pad(Float64.(x), pad, np - n - pad))
  S0 = sample(X .* φ, ts)
  S1 = zeros(spec.bins, length(f1))
  S2 = zeros(spec.bins, length(f2), length(f1))
  Threads.@threads for i in bands
    ψ = [abs(fk) < spec.fs / 2 ? morlet(fk, f1[i], σ1[i]) : 0.0 for fk in f]
    E = fft(abs.(ifft(X .* ψ)))
    S1[:, i] = sample(E .* φ, ts)
    for k in findall(active[:, i])
      E2 = fft(abs.(ifft(E .* morlet.(f, f2[k], σ2[k]))))
      S2[:, k, i] = sample(E2 .* φ, ts)
    end
  end
  S0, S1, S2
end
