using Metal

include(joinpath(@__DIR__, "..", "..", "test", "device.jl"))

@testset "Metal" begin
  device_tests(MtlArray)
end
