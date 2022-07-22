
abstract type AbstractPrimitive end

mutable struct Polytope{n,n3,T} <: AbstractPrimitive
    r::SVector{3,T}
    q::SVector{4,T}
    A::SMatrix{n,3,T,n3}
    b::SVector{n,T}
    function Polytope(A::SMatrix{n,3,T,n3}, b::SVector{n,T}) where{n,n3,T}
        new{n,n3,T}(SA[0,0,0.0],SA[1,0,0,0.0],A,b)
    end
end

mutable struct Capsule{T} <: AbstractPrimitive
    r::SVector{3,T}
    q::SVector{4,T}
    R::T
    L::T
    function Capsule(R::T,L::T) where {T}
        new{T}(SA[0,0,0.0],SA[1,0,0,0.0],R,L)
    end
end

mutable struct Cylinder{T} <: AbstractPrimitive
    r::SVector{3,T}
    q::SVector{4,T}
    R::T
    L::T
    function Cylinder(R::T,L::T) where {T}
        new{T}(SA[0,0,0.0],SA[1,0,0,0.0],R,L)
    end
end

mutable struct Cone{T} <: AbstractPrimitive
    r::SVector{3,T}
    q::SVector{4,T}
    H::T
    β::T
    function Cone(H::T,β::T) where {T}
        new{T}(SA[0,0,0.0],SA[1,0,0,0.0],H,β)
    end
end

mutable struct Sphere{T} <: AbstractPrimitive
	r::SVector{3,T}
    q::SVector{4,T} # not needed
    R::T
    function Sphere(R::T) where {T}
        new{T}(SA[0,0,0.0],SA[1,0,0,0.0],R)
    end
end

mutable struct Polygon{nh,nh2,T} <: AbstractPrimitive
	r::SVector{3,T}         # position in world frame
    q::SVector{4,T}         # quat for attitude (ᴺqᴮ)
    A::SMatrix{nh,2,T,nh2}  # polygon description (Ax≦b)
    b::SVector{nh,T}        # polygon description (Ax≦b)
    R::T                    # "cushion" radius
    function Polygon(A::SMatrix{nh,2,T,nh2},b::SVector{nh,T},R::T) where {nh,nh2,T}
        new{nh,nh2,T}(SA[0,0,0.0],SA[1,0,0,0.0],A,b,R)
    end
end
