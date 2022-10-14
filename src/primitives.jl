
abstract type AbstractPrimitive end

mutable struct Polytope{n,n3,T} <: AbstractPrimitive
    r::SVector{3,T}
    q::SVector{4,T}
    A::SMatrix{n,3,T,n3}
    b::SVector{n,T}
	r_offset::SVector{3,T}
	Q_offset::SMatrix{3,3,T,9}
    function Polytope(A::SMatrix{n,3,T,n3}, b::SVector{n,T}) where{n,n3,T}
        new{n,n3,T}(SA[0,0,0.0],SA[1,0,0,0.0],A,b,SA[0,0,0.0], SA[1 0 0; 0 1 0; 0 0 1.0])
    end
end

mutable struct Capsule{T} <: AbstractPrimitive
    r::SVector{3,T}
    q::SVector{4,T}
    R::T
    L::T
	r_offset::SVector{3,T}
	Q_offset::SMatrix{3,3,T,9}
    function Capsule(R::T,L::T) where {T}
        new{T}(SA[0,0,0.0],SA[1,0,0,0.0],R,L,SA[0,0,0.0], SA[1 0 0; 0 1 0; 0 0 1.0])
    end
end

mutable struct Cylinder{T} <: AbstractPrimitive
    r::SVector{3,T}
    q::SVector{4,T}
    R::T
    L::T
	r_offset::SVector{3,T}
	Q_offset::SMatrix{3,3,T,9}
    function Cylinder(R::T,L::T) where {T}
        new{T}(SA[0,0,0.0],SA[1,0,0,0.0],R,L,SA[0,0,0.0], SA[1 0 0; 0 1 0; 0 0 1.0])
    end
end

mutable struct Cone{T} <: AbstractPrimitive
    r::SVector{3,T}
    q::SVector{4,T}
    H::T
    β::T
	r_offset::SVector{3,T}
	Q_offset::SMatrix{3,3,T,9}
    function Cone(H::T,β::T) where {T}
        new{T}(SA[0,0,0.0],SA[1,0,0,0.0],H,β,SA[0,0,0.0], SA[1 0 0; 0 1 0; 0 0 1.0])
    end
end

mutable struct Sphere{T} <: AbstractPrimitive
	r::SVector{3,T}
    q::SVector{4,T} # not needed
    R::T
	r_offset::SVector{3,T}
	Q_offset::SMatrix{3,3,T,9}
    function Sphere(R::T) where {T}
        new{T}(SA[0,0,0.0],SA[1,0,0,0.0],R,SA[0,0,0.0], SA[1 0 0; 0 1 0; 0 0 1.0])
    end
end

mutable struct Polygon{nh,nh2,T} <: AbstractPrimitive
	r::SVector{3,T}         # position in world frame
    q::SVector{4,T}         # quat for attitude (ᴺqᴮ)
    A::SMatrix{nh,2,T,nh2}  # polygon description (Ax≦b)
    b::SVector{nh,T}        # polygon description (Ax≦b)
    R::T                    # "cushion" radius
	r_offset::SVector{3,T}
	Q_offset::SMatrix{3,3,T,9}
    function Polygon(A::SMatrix{nh,2,T,nh2},b::SVector{nh,T},R::T) where {nh,nh2,T}
        new{nh,nh2,T}(SA[0,0,0.0],SA[1,0,0,0.0],A,b,R,SA[0,0,0.0], SA[1 0 0; 0 1 0; 0 0 1.0])
    end
end

mutable struct Ellipsoid{T} <: AbstractPrimitive
	# x'*Q*x ≦ 1.0
	r::SVector{3,T}
    q::SVector{4,T}
    P::SMatrix{3,3,T,9}
	U::SMatrix{3,3,T,9}
	F::Eigen{T, T, SMatrix{3, 3, T, 9}, SVector{3, T}}
	r_offset::SVector{3,T}
	Q_offset::SMatrix{3,3,T,9}
    function Ellipsoid(P::SMatrix{3,3,T,9}) where {T}
        new{T}(
		SA[0,0,0.0],
		SA[1,0,0,0.0],
		P,
		SMatrix{3,3}(cholesky(P).U),
		eigen(P),
		SA[0,0,0.0],
		SA[1 0 0; 0 1 0; 0 0 1.0]
		)
    end
end
