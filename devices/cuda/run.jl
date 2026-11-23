# Test and time the package on an NVIDIA GPU, run from the package folder:
#   julia --project=devices/cuda devices/cuda/run.jl [gpu] [cpu]
#
# With no argument both parts run and everything, including any error, goes to
# devices/cuda/results/cuda.txt. With arguments only the parts named run, and
# the file is named after them, for example cuda_gpu.txt.

include(joinpath(@__DIR__, "select_gpu.jl"))

using CUDA, InteractiveUtils

include(joinpath(@__DIR__, "..", "..", "test", "device.jl"))
include(joinpath(@__DIR__, "..", "..", "benchmark", "common.jl"))
include(joinpath(@__DIR__, "..", "..", "benchmark", "sweep.jl"))

const PARTS = isempty(ARGS) ? ["gpu", "cpu"] : ARGS
const RESULT = joinpath(@__DIR__, "results", isempty(ARGS) ? "cuda.txt" : "cuda_$(join(ARGS, "_")).txt")

# run `f`, reporting an error instead of stopping
function attempt(f, title)
  println("\n## ", title)
  flush(stdout)
  try
    f()
  catch error
    showerror(stdout, error, catch_backtrace())
    println()
  end
  flush(stdout)
end

wait_for_gpu() = CUDA.synchronize()
free_gpu() = (GC.gc(); CUDA.reclaim())

function gpu_part()
  attempt(() -> device_tests(CuArray), "device tests")
  attempt("timings (csv)") do
    rows = benchmark("cuda", CuArray, Scattering1D.BatchedPlan; sync=wait_for_gpu, release=free_gpu)
    foreach(println, rows)
  end
  attempt("FFT length sets (csv)") do
    foreach(println, sweep(CuArray; sync=wait_for_gpu, release=free_gpu))
  end
end

function cpu_part(io)
  for threads in unique((1, Sys.CPU_THREADS ÷ 2, Sys.CPU_THREADS))
    attempt("timings on $threads CPU threads (csv)") do
      script = joinpath(@__DIR__, "..", "..", "benchmark", "cpu.jl")
      run(pipeline(`$(Base.julia_cmd()) --project=$(@__DIR__) -t $threads $script`; stdout=io, stderr=io))
    end
  end
end

mkpath(dirname(RESULT))
open(RESULT, "w") do io
  redirect_stdio(; stdout=io, stderr=io) do
    attempt(versioninfo, "Julia")
    attempt(CUDA.versioninfo, "CUDA")
    attempt(() -> println(CUDA.name(CUDA.device())), "GPU used")
    attempt(() -> run(`nvidia-smi`), "nvidia-smi")
    attempt(() -> run(`uptime`), "load before the run")
    "gpu" in PARTS && gpu_part()
    "cpu" in PARTS && cpu_part(io)
  end
end
println("wrote ", RESULT)
