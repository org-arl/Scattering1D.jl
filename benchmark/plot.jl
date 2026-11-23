# Draw the timings recorded on both machines:
#   julia benchmark/plot.jl
#
# Graphs are written as SVG, with PDF and PNG copies. `performance` is the one
# shown in the README. `comparison` adds the implementation this package
# replaces, and is drawn only where that implementation's timings are present.
# Each has four panels: machine (rows) by settings (columns). Colour
# tells the implementation and line style the device, so the same line means
# the same thing in every panel.

using Printf

const ROOT = joinpath(@__DIR__, "..")
const SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
const COLORS = (package="#2a78d6", old="#eb6834", kymatio="#1baf7a")
const DASHES = (gpu="", cpu="7 5", single="2 5")
const W, H, LEFT, RIGHT, TOP, GAP = 300, 260, 62, 58, 132, 84
const YMIN, YMAX = 0.03, 1000.0

struct Series
  label::String
  color::String
  dash::String
  points::Vector{Tuple{Int,Float64}}  # batch size, ms per signal
end

rows(path) = [split(line, ",") for line in readlines(path) if count(==(','), line) >= 4]

# timings of this package: lines of config,device,threads,batch,order,ms
function package(path, config, device, threads=nothing)
  [(parse(Int, r[4]), parse(Float64, r[6])) for r in rows(path)
   if length(r) == 6 && r[1] == config && r[2] == device && r[5] == "both" &&
      (isnothing(threads) || r[3] == string(threads))]
end

# timings of kymatio: lines of config,backend,threads,batch,ms
kymatio(path, config, backend) =
  [(parse(Int, r[4]), parse(Float64, r[5])) for r in rows(path) if r[1] == config && r[2] == backend]

# timings of the replaced implementation: lines of "timing batch ms"
function replaced(path)
  fields = [split(line) for line in readlines(path) if startswith(line, "timing ")]
  [(parse(Int, f[2]), parse(Float64, f[3])) for f in fields if length(f) == 3]
end

function panels(with_replaced)
  mac, remote = joinpath(ROOT, "benchmark", "results"), joinpath(ROOT, "devices", "cuda", "results")
  metal = joinpath(ROOT, "devices", "metal", "results", "metal.csv")
  both(results, cpu, gpu, config, kconfig, gpu_backend) = [
    Series("This package, CPU", COLORS.package, DASHES.cpu, cpu(config)),
    Series("This package, GPU", COLORS.package, DASHES.gpu, gpu(config)),
    Series("Kymatio, NumPy", COLORS.kymatio, DASHES.single, kymatio(results, kconfig, "numpy")),
    Series("Kymatio, PyTorch CPU", COLORS.kymatio, DASHES.cpu, kymatio(results, kconfig, "torch-cpu")),
    Series("Kymatio, PyTorch GPU", COLORS.kymatio, DASHES.gpu, kymatio(results, kconfig, gpu_backend)),
  ]
  mac_panel(config, kconfig) = both(joinpath(mac, "kymatio.csv"),
    c -> package(joinpath(mac, "cpu_12.csv"), c, "cpu"), c -> package(metal, c, "metal"), config, kconfig, "torch-mps")
  remote_panel(config, kconfig) = both(joinpath(remote, "kymatio.csv"),
    c -> package(joinpath(remote, "cuda.txt"), c, "cpu", 16),
    c -> package(joinpath(remote, "cuda_gpu.txt"), c, "cuda"), config, kconfig, "torch-cuda")
  old = Series("Replaced Julia code, GPU", COLORS.old, DASHES.gpu, replaced(joinpath(remote, "reference", "notes.txt")))
  [
    ("Apple M4 Max · constant-Q bank, Q = 35", mac_panel("working", "working")),
    ("Apple M4 Max · kymatio bank, J = 8, Q = 8", mac_panel("as-kymatio-kymatio", "kymatio")),
    ("Ryzen 9 5950X, RTX A6000 · constant-Q bank, Q = 35",
      vcat(remote_panel("working", "working"), with_replaced ? [old] : Series[])),
    ("Ryzen 9 5950X, RTX A6000 · kymatio bank, J = 8, Q = 8", remote_panel("as-kymatio-kymatio", "kymatio")),
  ]
end

xpos(batch) = log2(batch) / 7 * W
ypos(ms) = H - (log10(ms) - log10(YMIN)) / (log10(YMAX) - log10(YMIN)) * H

value(ms) = ms >= 100 ? @sprintf("%.0f", ms) : ms >= 10 ? @sprintf("%.1f", ms) : @sprintf("%.2f", ms)

# vertical positions for end labels, moved apart until none overlap
function label_positions(ys; spacing=13)
  order = sortperm(ys)
  placed = copy(ys)
  for _ in 1:50, (a, b) in zip(order[1:end-1], order[2:end])
    overlap = spacing - (placed[b] - placed[a])
    overlap > 0 || continue
    placed[a] -= overlap / 2
    placed[b] += overlap / 2
  end
  placed
end

text(io, x, y, content; size=11, fill=MUTED, extra="") =
  println(io, """<text x="$x" y="$y" font-size="$size" fill="$fill" $extra>$content</text>""")

function draw_panel(io, title, series, x0, y0)
  println(io, """<g transform="translate($x0,$y0)">""")
  text(io, 0, -14, title; size=12.5, fill=INK, extra="font-weight=\"600\"")
  for ms in (0.1, 1, 10, 100, 1000)
    println(io, """<line x1="0" x2="$W" y1="$(ypos(ms))" y2="$(ypos(ms))" stroke="$GRID"/>""")
    text(io, -8, ypos(ms) + 4, ms < 1 ? ms : Int(ms); extra="text-anchor=\"end\"")
  end
  for batch in (1, 2, 4, 8, 16, 32, 64, 128)
    text(io, xpos(batch), H + 18, batch; extra="text-anchor=\"middle\"")
  end
  println(io, """<line x1="0" x2="$W" y1="$H" y2="$H" stroke="$MUTED"/>""")
  text(io, W / 2, H + 38, "batch size"; size=12, extra="text-anchor=\"middle\"")
  text(io, 0, 0, "ms per signal"; size=12, extra="transform=\"translate(-42,$(H / 2)) rotate(-90)\" text-anchor=\"middle\"")
  for s in series
    path = join((@sprintf("%s%.1f,%.1f", i == 1 ? "M" : "L", xpos(b), ypos(ms)) for (i, (b, ms)) in enumerate(s.points)), " ")
    dash = isempty(s.dash) ? "" : """ stroke-dasharray="$(s.dash)\""""
    println(io, """<path d="$path" fill="none" stroke="$(s.color)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"$dash/>""")
    for (b, ms) in s.points
      println(io, """<circle cx="$(xpos(b))" cy="$(ypos(ms))" r="4" fill="$(s.color)" stroke="$SURFACE" stroke-width="2"><title>$(s.label), batch $b: $(value(ms)) ms per signal</title></circle>""")
    end
  end
  ends = [ypos(s.points[end][2]) for s in series]
  for (s, y, yl) in zip(series, ends, label_positions(ends))
    x = xpos(s.points[end][1])
    println(io, """<line x1="$(x + 7)" x2="$(W + 10)" y1="$y" y2="$yl" stroke="$GRID"/>""")
    text(io, W + 13, yl + 4, value(s.points[end][2]); fill=INK)
  end
  println(io, "</g>")
end

function legend(io, x, y, with_replaced)
  entry(x, y, color, dash, label) = begin
    d = isempty(dash) ? "" : """ stroke-dasharray="$dash\""""
    println(io, """<line x1="$x" x2="$(x + 34)" y1="$y" y2="$y" stroke="$color" stroke-width="2"$d/>""")
    text(io, x + 42, y + 4, label; size=12, fill=INK)
  end
  entry(x, y, COLORS.package, "", "This package")
  entry(x + 150, y, COLORS.kymatio, "", "Kymatio")
  with_replaced && entry(x + 270, y, COLORS.old, "", "Replaced Julia code (fails at batch 128)")
  entry(x, y + 22, MUTED, DASHES.gpu, "GPU")
  entry(x + 150, y + 22, MUTED, DASHES.cpu, "CPU, all cores")
  entry(x + 330, y + 22, MUTED, DASHES.single, "CPU, one thread (NumPy)")
end

function draw(name, with_replaced)
  width, height = 2 * (LEFT + W + RIGHT), TOP + 2 * (H + GAP)
  svg = joinpath(@__DIR__, "$name.svg")
  open(svg, "w") do io
    println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height" font-family="-apple-system, 'Helvetica Neue', Arial, sans-serif">""")
    println(io, """<rect width="$width" height="$height" fill="$SURFACE"/>""")
    text(io, LEFT, 28, "Scattering transform: time per 1 s signal against batch size"; size=16, fill=INK, extra="font-weight=\"600\"")
    text(io, LEFT, 48, "Signals of 40,000 samples. Both axes are logarithmic; lower is faster. Line ends show ms per signal."; size=12)
    text(io, LEFT, 64, "For the constant-Q bank, kymatio runs its closest available setting, which is not identical work."; size=12)
    legend(io, LEFT, 86, with_replaced)
    for (i, (title, series)) in enumerate(panels(with_replaced))
      col, row = (i - 1) % 2, (i - 1) ÷ 2
      draw_panel(io, title, filter(s -> !isempty(s.points), series), LEFT + col * (LEFT + W + RIGHT), TOP + 30 + row * (H + GAP))
    end
    println(io, "</svg>")
  end
  # PDF and PNG copies, made with rsvg-convert where it is installed
  isnothing(Sys.which("rsvg-convert")) && return
  run(`rsvg-convert --format=pdf --output=$(joinpath(@__DIR__, "$name.pdf")) $svg`)
  run(`rsvg-convert --format=png --zoom=2 --output=$(joinpath(@__DIR__, "$name.png")) $svg`)
end

draw("performance", false)
isfile(joinpath(ROOT, "devices", "cuda", "results", "reference", "notes.txt")) && draw("comparison", true)
