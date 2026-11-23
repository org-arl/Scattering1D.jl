# Reverse pass of the batched engine. A step that read rows through an index
# is undone by reading back through the inverse index, one term per reader of
# a row, so that no two writes ever land on the same entry.

### types

# What the reverse pass adds to one `Group`.
struct ReverseGroup{F,G,H,PS,PE,PO}
  unfold::Vector{F}    # output-spectrum gradient to envelope-spectrum gradient
  ungather::Vector{G}  # gradient of the filters' windows to that of the spectrum they read
  halve::H             # weights that turn a half-spectrum gradient into what the inverse FFT expects
  spread::PS
  envelope::PE
  output::PO
end

struct BatchedReverse{F,G,H,Z2,RG<:ReverseGroup,PO,PX}
  unfold::Vector{F}     # for the low-passed signal
  unreflect::Vector{G}  # padded-signal gradient to signal gradient
  halve::H              # for the padded signal's spectrum
  double::H             # for an output spectrum
  Ē::Z2                 # gradient of every band's envelope spectrum
  X̄::Z2                 # gradient of the padded signals' spectra
  groups1::Vector{RG}
  groups2::Vector{RG}
  output::PO
  signal::PX
end

### public API

function backpropagate!(x̄::AbstractMatrix, plan::BatchedPlan, x::AbstractMatrix, c̄)
  check_sizes(plan, x, c̄.S0, c̄.S1)
  size(x̄) == size(x) || throw(DimensionMismatch("gradient and signals differ in size"))
  S̄2 = second_order(c̄)
  isnothing(S̄2) || size(S̄2) == (plan.geometry.bins, plan.slots, plan.bands, plan.batch) ||
    throw(DimensionMismatch("S2 does not match the plan"))
  isnothing(plan.reverse[]) && (plan.reverse[] = BatchedReverse(plan))
  reverse!(x̄, plan, plan.reverse[], x, c̄.S0, c̄.S1, S̄2)
  x̄
end

### helpers

function BatchedReverse(plan::BatchedPlan{T}) where T
  g = plan.geometry
  on(a) = copyto!(similar(plan.x, eltype(a), size(a)), a)
  terms(args...) = [Fold(on.(t)...) for t in inverse_terms(args...)]
  half = g.m ÷ 2 + 1
  double = on(T[k == 1 || (iseven(g.m) && k == half) ? 1 : 2 for k in 1:half])
  reverse_group(group, nrows) = begin
    M, K = size(group.Z, 1), size(group.E, 1)
    src, w, c = Array(group.src), Array(group.w), Array(group.c)
    ReverseGroup(terms(unfolded(group.folds, half)..., K), terms(eachindex(src), src, w, c, nrows),
      on(halving(T, M)), plan_fft!(group.Z, 1), plan_brfft(group.E, M, 1), plan_rfft(group.R, 1))
  end
  reflect = Array(plan.reflect)
  BatchedReverse(terms(unfolded(plan.folds, half)..., size(plan.X, 1)),
    terms(eachindex(reflect), reflect, ones(T, g.np), falses(g.np), g.n), on(halving(T, g.np)), double,
    similar(plan.E), similar(plan.X), [reverse_group(group, size(plan.X, 1)) for group in plan.groups1],
    [reverse_group(group, size(plan.E, 1)) for group in plan.groups2],
    plan_rfft(plan.R, 1), plan_brfft(plan.X, g.np, 1))
end

# weights that halve the bins strictly between 0 and the Nyquist frequency of
# a spectrum of `M` samples
halving(::Type{T}, M) where T = T[k == 1 || (iseven(M) && k == M ÷ 2 + 1) ? 1 : 1 // 2 for k in 1:M÷2+1]

# the reads of a low-pass sum as lists: output bin, envelope bin, weight, conjugated
function unfolded(folds, half)
  src, w, c = (reduce(vcat, (Array(getfield(f, name)) for f in folds)) for name in (:src, :w, :c))
  repeat(1:half, length(folds)), src, w, c
end

# Terms that undo a gather in which entry `entries[i]` read row `rows[i]` of
# `nrows` with weight `w[i]`, conjugated where `c[i]`. Each term gives, for
# every row, one entry that read it, the weight to apply and whether to
# conjugate. Rows read by several entries get one term each.
function inverse_terms(entries, rows, w, c, nrows)
  terms = Tuple{Vector{Int32},Vector{eltype(w)},Vector{Bool}}[]
  readers = zeros(Int, nrows)
  for i in eachindex(rows)
    iszero(w[i]) && continue
    t = (readers[rows[i]] += 1)
    t > length(terms) && push!(terms, (ones(Int32, nrows), zeros(eltype(w), nrows), zeros(Bool, nrows)))
    terms[t][1][rows[i]], terms[t][3][rows[i]] = entries[i], c[i]
    terms[t][2][rows[i]] = c[i] ? w[i] : conj(w[i])
  end
  terms
end

# gradient of a signal whose magnitude times `scale` has gradient `ē`
function through_magnitude(z, ē, scale)
  a = sqrt(abs2(z))
  z * ifelse(a > 0, ē * scale / a, zero(a))
end

# Ḡ .+= the entries of `Ȳ` that read each row of the gathered array
function ungather!(Ḡ, Ȳ, terms)
  for t in terms
    Ḡ .+= pick.(term(Ȳ, t), t.w, t.c)
  end
end

function reverse!(x̄, plan::BatchedPlan{T}, reverse::BatchedReverse, x, S̄0, S̄1, S̄2) where T
  g = plan.geometry
  kept = g.row+1:g.row+g.bins
  plan.x .= view(x, plan.reflect, :)
  mul!(plan.X, plan.forward, plan.x)
  for group in plan.groups1
    group.Zr .= pick.(view(plan.X, group.src, :), group.w, group.c)
    group.inverse * group.Z
    isnothing(S̄2) && continue
    group.e .= magnitude.(group.Z, convert(T, group.scale))
    mul!(group.E, group.envelope, group.e)
    view(plan.E, group.rows, :) .= group.Er
  end
  fill!(reverse.Ē, 0)
  if !isnothing(S̄2)
    S̄2r = reshape(S̄2, g.bins, :, plan.batch)
    for (group, r) in zip(plan.groups2, reverse.groups2)
      group.Zr .= pick.(view(plan.E, group.src, :), group.w, group.c)
      group.inverse * group.Z
      fill!(group.E, 0)
      group_pullback!(reverse.Ē, group, r, S̄2r, reverse.double, kept, T)
    end
  end
  fill!(reverse.X̄, 0)
  for (group, r) in zip(plan.groups1, reverse.groups1)
    group.Er .= view(reverse.Ē, group.rows, :)
    group_pullback!(reverse.X̄, group, r, S̄1, reverse.double, kept, T)
  end
  fill!(plan.R, 0)
  view(plan.R, kept, :) .= S̄0
  mul!(plan.F, reverse.output, plan.R)
  plan.F .*= reverse.double
  ungather!(reverse.X̄, plan.F, reverse.unfold)
  reverse.X̄ .*= reverse.halve
  mul!(plan.x, reverse.signal, reverse.X̄)
  fill!(x̄, 0)
  ungather!(x̄, plan.x, reverse.unreflect)
end

# Add to `Ḡ` the gradient of the spectrum that `group` read, given the
# gradient `S̄` of its output samples. `group.Z` holds the group's signals and
# `group.E` any gradient of their envelope spectra that arrived from elsewhere.
function group_pullback!(Ḡ, group, r, S̄, double, kept, ::Type{T}) where T
  fill!(group.R, 0)
  view(group.R, kept, :, :) .= view(S̄, :, group.cols, :)
  mul!(group.F, r.output, group.R)
  group.F .*= double
  ungather!(group.E, group.F, r.unfold)
  group.E .*= r.halve
  mul!(group.e, r.envelope, group.E)
  group.Z .= through_magnitude.(group.Z, group.e, convert(T, group.scale))
  r.spread * group.Z
  ungather!(Ḡ, group.Zr, r.ungather)
end
