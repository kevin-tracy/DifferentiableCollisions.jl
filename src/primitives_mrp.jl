
abstract type AbstractPrimitiveMRP end

mutable struct PolytopeMRP{n,n3,T} <: AbstractPrimitiveMRP
    r::SVector{3,T}
    p::SVector{3,T}
    A::SMatrix{n,3,T,n3}
    b::SVector{n,T}
    function Polytope(A::SMatrix{n,3,T,n3}, b::SVector{n,T}) where{n,n3,T}
        new{n,n3,T}(SA[0,0,0.0],SA[0,0,0.0],A,b)
    end
end

mutable struct CapsuleMRP{T} <: AbstractPrimitiveMRP
    r::SVector{3,T}
    p::SVector{3,T}
    R::T
    L::T
    function Capsule(R::T,L::T) where {T}
        new{T}(SA[0,0,0.0],SA[0,0,0.0],R,L)
    end
end

mutable struct CylinderMRP{T} <: AbstractPrimitiveMRP
    r::SVector{3,T}
    p::SVector{3,T}
    R::T
    L::T
    function Cylinder(R::T,L::T) where {T}
        new{T}(SA[0,0,0.0],SA[0,0,0.0],R,L)
    end
end

mutable struct ConeMRP{T} <: AbstractPrimitiveMRP
    r::SVector{3,T}
    p::SVector{3,T}
    H::T
    β::T
    function Cone(H::T,β::T) where {T}
        new{T}(SA[0,0,0.0],SA[0,0,0.0],H,β)
    end
end

mutable struct SphereMRP{T} <: AbstractPrimitiveMRP
	r::SVector{3,T}
    p::SVector{3,T} # not needed
    R::T
    function Sphere(R::T) where {T}
        new{T}(SA[0,0,0.0],SA[0,0,0.0],R)
    end
end

mutable struct PolygonMRP{nh,nh2,T} <: AbstractPrimitiveMRP
	r::SVector{3,T}         # position in world frame
    p::SVector{3,T}         # quat for attitude (ᴺqᴮ)
    A::SMatrix{nh,2,T,nh2}  # polygon description (Ax≦b)
    b::SVector{nh,T}        # polygon description (Ax≦b)
    R::T                    # "cushion" radius
    function Polygon(A::SMatrix{nh,2,T,nh2},b::SVector{nh,T},R::T) where {nh,nh2,T}
        new{nh,nh2,T}(SA[0,0,0.0],SA[0,0,0.0],A,b,R)
    end
end
