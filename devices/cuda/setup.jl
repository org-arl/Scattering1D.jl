# One-time setup of the Julia environment used on the NVIDIA machine, run from
# the package folder:
#   julia --project=devices/cuda devices/cuda/setup.jl

using Pkg

Pkg.develop(path=joinpath(@__DIR__, "..", ".."))
Pkg.add(["CUDA", "FFTW"])
Pkg.add(["Test", "Random", "Printf", "Statistics", "LinearAlgebra", "InteractiveUtils"])
