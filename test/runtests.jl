using Test
using LinearAlgebra
using StaticArrays
using BenchmarkTools
using SparseArrays
using JLD2

import DCD
import FiniteDiff
import ForwardDiff
import Random
Random.seed!(1234)

@testset "NT scaling" begin
    include("nt_scaling_chol_tests.jl")
    include("nt_scaling_chol_2_tests.jl")
end

@testset "soc utils" begin
    include("soc_utils_tests.jl")
end

@testset "solvers" begin
    include("solver_tests.jl")
end
#
# @testset "derivatives" begin
#     include("deriv_tests.jl")
# end
@testset "derivatives" begin
    include("deriv_tests_2.jl")
end
