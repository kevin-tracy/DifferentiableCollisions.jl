__precompile__(true)
module DCOL

using LinearAlgebra
using StaticArrays
using Printf
import Polyhedra
import MeshCat as mc
import ForwardDiff

# solver stuff
include("solvers/coneqp/NT_scaling_chol_2.jl")
include("solvers/coneqp/soc_utils.jl")
include("solvers/coneqp/static_solver2.jl")

# primitives
include("primitives.jl")
include("primitives_mrp.jl")
include("mass_properties.jl")

# visualizer
include("visualizer.jl")

# proximity stuff
include("problem_matrices.jl")
include("combine_problem_matrices.jl")
include("proximity.jl")
include("proximity_gradient.jl")

#
include("misc_primitive_constructors.jl")



export *, \

end # module
