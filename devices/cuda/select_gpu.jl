# Restrict CUDA to the GPU with the most memory, unless the caller has already
# chosen with CUDA_VISIBLE_DEVICES. Must be included before CUDA is loaded.
# The GPU is named by its UUID, which does not depend on how GPUs are numbered.

largest_gpu(listing) = first(argmax(row -> parse(Int, row[2]), split.(split(strip(listing), "\n"), ", ")))

if !haskey(ENV, "CUDA_VISIBLE_DEVICES")
  query = `nvidia-smi --query-gpu=uuid,memory.total --format=csv,noheader,nounits`
  ENV["CUDA_VISIBLE_DEVICES"] = largest_gpu(read(query, String))
end
