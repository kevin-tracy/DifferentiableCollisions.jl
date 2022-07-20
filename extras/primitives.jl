cd("/Users/kevintracy/.julia/dev/DCD/extras")
import Pkg; Pkg.activate(".")

using StaticArrays, Polyhedra, MeshCat, LinearAlgebra, JLD2
mutable struct Polytope{n,n3,T}
    r::SVector{3,T}
    q::SVector{4,T}
    A::SMatrix{n,3,T,n3}
    b::SVector{n,T}
    function Polytope(A::SMatrix{n,3,T,n3}, b::SVector{n,T}) where{n,n3,T}
        new{n,n3,T}(SA[0,0,0.0],SA[1,0,0,0.0],A,b)
    end
end

let

    @load "polytopes.jld2"

    # @show size(A2)
    polytope = Polytope(SMatrix{8,3}(A2),SVector{8}(b2))

end
