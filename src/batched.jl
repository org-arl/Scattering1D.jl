### constants

# FFT lengths offered to the batched engine are these times a power of two,
# over four. Powers of two alone give the fewest groups of filters, which was
# fastest on both an Apple M4 Max and an NVIDIA RTX A6000 although the signals
# get longer: each call to the device has a cost that more lengths multiply.
const MANTISSAS = (8,)

### types

# One term of the sum that low-passes envelope spectra and aliases them onto
# the output bins.
struct Fold{AI,AC,AB}
  src::AI   # envelope bin read for each output bin
  w::AC     # low-pass weight, zero where the term has nothing to add
  c::AB     # whether the bin read is conjugated
end

# All filters of one order that share an FFT length, across the whole batch.
struct Group{AI,AR,AB,AC,Z2,Z3,R3,PZ,PE,PF}
  src::AI               # spectrum row read for each window entry of each filter
  w::AR                 # filter weights, zero beyond each window
  c::AB                 # whether the row read is conjugated
  scale::Float64        # factor applied to the envelope
  rows::UnitRange{Int}  # where a band group's envelope spectra go in `BatchedPlan.E`
  cols::AI              # output column of each filter
  folds::Vector{Fold{AI,AC,AB}}
  Zr::Z2                # `Z` seen as (sample, filter) × batch
  Z::Z3                 # band or path signals, sample × filter × batch
  e::R3                 # envelopes
  Er::Z2                # `E` seen as (bin, filter) × batch
  E::Z3                 # envelope spectra
  F::Z3                 # spectra of the output samples
  R::R3                 # output samples spanning the padded signal
  inverse::PZ
  envelope::PE
  output::PF
end

# Plan for any array type that has broadcasting, indexed views and FFTs along
# the first dimension. Filters with equal FFT lengths run as one transform.
struct BatchedPlan{T,AI,AC,AB,R2,Z2,G<:Group,PX,PF} <: ScatteringPlan{T}
  geometry::Geometry
  slots::Int
  bands::Int
  batch::Int
  reflect::AI           # signal sample at each padded sample
  folds::Vector{Fold{AI,AC,AB}}
  x::R2                 # padded signals
  X::Z2                 # their spectra
  E::Z2                 # envelope spectra of every band, one after another
  F::Z2
  R::R2
  groups1::Vector{G}
  groups2::Vector{G}
  forward::PX
  output::PF
  reverse::Base.RefValue{Any}  # plans and tables of the reverse pass, built at its first use
end

### public API

function ScatteringPlan(spec::ScatteringSpec, x::AbstractMatrix{T}) where T<:AbstractFloat
  BatchedPlan(Tables{T}(spec, size(x, 1), coarse_size), x)
end

function scattering!(c::Coefficients, plan::BatchedPlan, x::AbstractMatrix)
  check_sizes(plan, x, c.S0, c.S1)
  size(c.S2) == (plan.geometry.bins, plan.slots, plan.bands, plan.batch) ||
    throw(DimensionMismatch("S2 does not match the plan"))
  transform!(plan, x, c.S0, c.S1, c.S2)
  c
end

function first_order!(c::FirstOrder, plan::BatchedPlan, x::AbstractMatrix)
  check_sizes(plan, x, c.S0, c.S1)
  transform!(plan, x, c.S0, c.S1, nothing)
  c
end

### helpers

function BatchedPlan(t::Tables{T}, x::AbstractMatrix) where T
  g, batch = t.geometry, size(x, 2)
  on(a) = copyto!(similar(x, eltype(a), size(a)), a)
  kφ, half = length(t.φ) - 1, g.m ÷ 2 + 1
  reflect = on(Int32[reflected(i, g) for i in 1:g.np])
  xp, R = similar(x, T, g.np, batch), similar(x, T, g.m, batch)
  X, F = similar(x, Complex{T}, g.np ÷ 2 + 1, batch), similar(x, Complex{T}, half, batch)
  folds = [Fold(on.(f)...) for f in fold_terms(t.φ, g.m, min(kφ, (g.np - 1) ÷ 2), 1 / g.np)]
  # rows of each band's envelope spectrum in the list shared by all bands
  order = sortperm([b.M for b in t.bands])
  counts = [b.M ÷ 2 + 1 for b in t.bands[order]]
  starts = zeros(Int, length(order))
  starts[order] = cumsum(counts) .- counts
  E = similar(x, Complex{T}, sum(counts), batch)
  groups1 = map(unique(b.M for b in t.bands[order])) do M
    members = [i for i in order if t.bands[i].M == M]
    stages = [(t.bands[i].k0, t.bands[i].at, t.bands[i].len, g.np, 0) for i in members]
    Group(x, t, M, stages, t.w1, 1 / M, 1.0, starts[members[1]]+1:starts[members[end]]+M÷2+1, members)
  end
  groups2 = map(sort!(unique(p.M for p in t.paths))) do M
    members = [p for p in t.paths if p.M == M]
    stages = [(p.k0, p.at, p.len, t.bands[p.band].M, starts[p.band]) for p in members]
    Group(x, t, M, stages, t.w2, 1.0, 1 / M, 1:0, [p.slot + t.slots * (p.band - 1) for p in members])
  end
  forward, output = plan_rfft(xp, 1), plan_brfft(F, g.m, 1)
  BatchedPlan{T,typeof(reflect),typeof(folds[1].w),typeof(folds[1].c),typeof(xp),typeof(X),
    eltype(groups1),typeof(forward),typeof(output)}(g, t.slots, length(t.bands), batch, reflect,
    folds, xp, X, E, F, R, groups1, groups2, forward, output, Ref{Any}(nothing))
end

# signal sample that reflection puts at padded sample `t`
reflected(t, g::Geometry) = t <= g.pad ? g.pad - t + 2 : t <= g.pad + g.n ? t - g.pad : 2g.n + g.pad - t

# Group of the filters `stages`, each (first bin, first weight, weights,
# period of the spectrum read, rows before that spectrum).
function Group(x::AbstractMatrix{T}, t::Tables, M, stages, w, scale, fold_scale, rows, cols) where T
  g, batch, count = t.geometry, size(x, 2), length(stages)
  on(a) = copyto!(similar(x, eltype(a), size(a)), a)
  src, weights, c = ones(Int32, M, count), zeros(T, M, count), zeros(Bool, M, count)
  for (j, (k0, at, len, period, before)) in enumerate(stages), d in 0:len-1
    bin = mod(k0 + d, period)
    c[d + 1, j] = bin > period ÷ 2
    src[d + 1, j] = before + (c[d + 1, j] ? period - bin : bin) + 1
    weights[d + 1, j] = w[at + d]
  end
  kφ = length(t.φ) - 1
  folds = [Fold(on.(f)...) for f in fold_terms(t.φ, g.m, min(kφ, (M - 1) ÷ 2), fold_scale)]
  Z, e = similar(x, Complex{T}, M, count, batch), similar(x, T, M, count, batch)
  E = similar(x, Complex{T}, M ÷ 2 + 1, count, batch)
  F, R = similar(x, Complex{T}, g.m ÷ 2 + 1, count, batch), similar(x, T, g.m, count, batch)
  Group(on(vec(src)), on(vec(weights)), on(vec(c)), Float64(scale), rows, on(Int32.(cols)), folds,
    reshape(Z, M * count, batch), Z, e, reshape(E, :, batch), E, F, R, plan_bfft!(Z, 1),
    plan_rfft(e, 1), plan_brfft(F, g.m, 1))
end

# Terms (src, w, c) whose sum over terms gives, for each output bin, the
# low-passed envelope bins 0 to kmax that alias onto it.
function fold_terms(φ::Vector{Complex{T}}, m, kmax, scale) where T
  half = m ÷ 2
  terms = Tuple{Vector{Int32},Vector{Complex{T}},Vector{Bool}}[]
  function add!(bin, k, conjugated)
    q = findfirst(term -> term[2][bin + 1] == 0, terms)
    if isnothing(q)
      push!(terms, (ones(Int32, half + 1), zeros(Complex{T}, half + 1), zeros(Bool, half + 1)))
      q = length(terms)
    end
    terms[q][1][bin + 1], terms[q][3][bin + 1] = k + 1, conjugated
    terms[q][2][bin + 1] = (conjugated ? conj(φ[k + 1]) : φ[k + 1]) * scale
  end
  for k in 0:kmax
    mod(k, m) <= half && add!(mod(k, m), k, false)
    k > 0 && mod(-k, m) <= half && add!(mod(-k, m), k, true)
  end
  terms
end

function coarse_size(n, mantissas=MANTISSAS)
  for octave in 0:62, mantissa in mantissas
    size = (mantissa << octave) ÷ 4
    size >= n && iseven(size) && return size
  end
end

dimensions(plan::BatchedPlan) = (plan.geometry.bins, plan.slots, plan.bands)
Base.zeros(plan::BatchedPlan{T}, dims...) where T = fill!(similar(plan.x, T, dims...), 0)

function check_sizes(plan::BatchedPlan, x, S0, S1)
  g = plan.geometry
  size(x) == (g.n, plan.batch) ||
    throw(DimensionMismatch("plan is for $(g.n) samples × $(plan.batch) signals"))
  size(S0) == (g.bins, plan.batch) && size(S1) == (g.bins, plan.bands, plan.batch) ||
    throw(DimensionMismatch("coefficients do not match the plan"))
end

pick(x, w, c) = ifelse(c, conj(x), x) * w

magnitude(z, scale) = sqrt(abs2(z)) * scale

# bins of `E` read by one term of the low-pass sum
term(E, fold) = view(E, fold.src, ntuple(_ -> :, ndims(E) - 1)...)

# The usual one or two terms are written in one pass over the output.
function lowpass!(F, E, folds)
  a = folds[1]
  if length(folds) == 1
    F .= pick.(term(E, a), a.w, a.c)
    return
  end
  b = folds[2]
  F .= pick.(term(E, a), a.w, a.c) .+ pick.(term(E, b), b.w, b.c)
  for q in 3:length(folds)
    F .+= pick.(term(E, folds[q]), folds[q].w, folds[q].c)
  end
end

function transform!(plan::BatchedPlan{T}, x, S0, S1, S2) where T
  g = plan.geometry
  kept = g.row+1:g.row+g.bins
  plan.x .= view(x, plan.reflect, :)
  mul!(plan.X, plan.forward, plan.x)
  lowpass!(plan.F, plan.X, plan.folds)
  mul!(plan.R, plan.output, plan.F)
  S0 .= view(plan.R, kept, :)
  for group in plan.groups1
    envelopes!(group, plan.X, T)
    view(plan.E, group.rows, :) .= group.Er
    emit!(S1, group, kept)
  end
  isnothing(S2) && return
  S2r = reshape(S2, g.bins, :, plan.batch)
  for group in plan.groups2
    envelopes!(group, plan.E, T)
    emit!(S2r, group, kept)
  end
end

function envelopes!(group, X, ::Type{T}) where T
  group.Zr .= pick.(view(X, group.src, :), group.w, group.c)
  group.inverse * group.Z
  group.e .= magnitude.(group.Z, convert(T, group.scale))
  mul!(group.E, group.envelope, group.e)
end

function emit!(S, group, kept)
  lowpass!(group.F, group.E, group.folds)
  mul!(group.R, group.output, group.F)
  view(S, :, group.cols, :) .= view(group.R, kept, :, :)
end
