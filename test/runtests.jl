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
    # include("nt_scaling_chol_tests.jl")
    include("nt_scaling_chol_2_tests.jl")
end

@testset "soc utils" begin
    include("soc_utils_tests.jl")
end

@testset "socp solvers" begin
    include("solver_tests.jl")
end

@testset "lp solver" begin
    include("lp_solver_tests.jl")
end

@testset "derivatives" begin
    include("proximity_test.jl")
end

@testset "combine matrices" begin
    include("combine_matrices_test.jl")
end

@testset "polytope derivs" begin
    include("polytope_derivs_test.jl")
end
