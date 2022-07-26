__precompile__(true)
module DCD

using LinearAlgebra
using StaticArrays
using Printf
using BenchmarkTools
import MeshCat as mc
import ForwardDiff

include("coneqp/nt_scaling_chol.jl")
include("coneqp/nt_scaling_chol_2.jl")
include("coneqp/soc_utils.jl")
include("coneqp/static_solver.jl")
include("coneqp/static_solver2.jl")
include("primitives.jl")
include("visualizer.jl")
include("problem_matrices.jl")
include("combine_problem_matrices.jl")
include("proximity.jl")

export *, \

end # module
