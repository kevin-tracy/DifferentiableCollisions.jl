__precompile__(true)
module DCD

using LinearAlgebra
using StaticArrays
using Printf
using BenchmarkTools

# include("coneqp/nt_scaling_lite.jl")
# include("coneqp/nt_scaling.jl")
include("coneqp/nt_scaling_chol.jl")
include("coneqp/nt_scaling_chol_2.jl")
include("coneqp/soc_utils.jl")
include("coneqp/static_solver.jl")
include("coneqp/static_solver2.jl")

export *, \

end # module
