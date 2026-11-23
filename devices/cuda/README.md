# Runs on the NVIDIA machine

All commands are run from the package folder. Make the machine quiet first: other jobs on the CPU or the GPU spoil the timings.

## Choosing the GPU

Every script restricts itself to the GPU with the most memory, which on a machine with an RTX A6000 and a smaller card is the A6000, and records the name of the GPU it used. To choose another, set `CUDA_VISIBLE_DEVICES` before the run.

## Setup, once

```
julia --project=devices/cuda devices/cuda/setup.jl
sh benchmark/setup_kymatio.sh
```

## Runs

| Command | Result to bring back |
|---|---|
| `julia --project=devices/cuda devices/cuda/run.jl` | `devices/cuda/results/cuda.txt` |
| `julia --project=devices/cuda devices/cuda/run.jl gpu` | `devices/cuda/results/cuda_gpu.txt` |
| `benchmark/.venv/bin/python benchmark/kymatio_bench.py` | `benchmark/results/kymatio.csv` |

`run.jl` tests this package on the GPU, then times it on the GPU and on CPU threads; with `gpu` or `cpu` as an argument it runs only that part. `kymatio_bench.py` times kymatio. Each catches its own errors and records them in its result, so a failed step still produces a file worth bringing back.
