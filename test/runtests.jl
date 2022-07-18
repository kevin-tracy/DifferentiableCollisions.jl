using Test
using LinearAlgebra
using StaticArrays
using BenchmarkTools
using SparseArrays

import DCD


@testset "NT scaling" begin
    include("nt_scaling_tests.jl")
end

@testset "soc utils" begin
    include("soc_utils_tests.jl")
end
