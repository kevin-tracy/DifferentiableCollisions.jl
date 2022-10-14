
abstract type AbstractPrimitiveMRP end

mutable struct PolytopeMRP{n,n3,T} <: AbstractPrimitiveMRP
    r::SVector{3,T}
    p::SVector{3,T}
    A::SMatrix{n,3,T,n3}
    b::SVector{n,T}
	r_offset::SVector{3,T}
	Q_offset::SMatrix{3,3,T,9}
    function PolytopeMRP(A::SMatrix{n,3,T,n3}, b::SVector{n,T}) where{n,n3,T}
        new{n,n3,T}(SA[0,0,0.0],SA[0,0,0.0],A,b,SA[0,0,0.0], SA[1 0 0; 0 1 0; 0 0 1.0])
    end
end

mutable struct CapsuleMRP{T} <: AbstractPrimitiveMRP
    r::SVector{3,T}
    p::SVector{3,T}
    R::T
    L::T
	r_offset::SVector{3,T}
	Q_offset::SMatrix{3,3,T,9}
    function CapsuleMRP(R::T,L::T) where {T}
        new{T}(SA[0,0,0.0],SA[0,0,0.0],R,L,SA[0,0,0.0], SA[1 0 0; 0 1 0; 0 0 1.0])
    end
end

mutable struct CylinderMRP{T} <: AbstractPrimitiveMRP
    r::SVector{3,T}
    p::SVector{3,T}
    R::T
    L::T
	r_offset::SVector{3,T}
	Q_offset::SMatrix{3,3,T,9}
    function CylinderMRP(R::T,L::T) where {T}
        new{T}(SA[0,0,0.0],SA[0,0,0.0],R,L,SA[0,0,0.0], SA[1 0 0; 0 1 0; 0 0 1.0])
    end
end

mutable struct ConeMRP{T} <: AbstractPrimitiveMRP
    r::SVector{3,T}
    p::SVector{3,T}
    H::T
    β::T
	r_offset::SVector{3,T}
	Q_offset::SMatrix{3,3,T,9}
    function ConeMRP(H::T,β::T) where {T}
        new{T}(SA[0,0,0.0],SA[0,0,0.0],H,β,SA[0,0,0.0], SA[1 0 0; 0 1 0; 0 0 1.0])
    end
end

mutable struct SphereMRP{T} <: AbstractPrimitiveMRP
	r::SVector{3,T}
    p::SVector{3,T} # not needed
    R::T
	r_offset::SVector{3,T}
	Q_offset::SMatrix{3,3,T,9}
    function SphereMRP(R::T) where {T}
        new{T}(SA[0,0,0.0],SA[0,0,0.0],R,SA[0,0,0.0], SA[1 0 0; 0 1 0; 0 0 1.0])
    end
end

mutable struct PolygonMRP{nh,nh2,T} <: AbstractPrimitiveMRP
	r::SVector{3,T}         # position in world frame
    p::SVector{3,T}         # quat for attitude (ᴺqᴮ)
    A::SMatrix{nh,2,T,nh2}  # polygon description (Ax≦b)
    b::SVector{nh,T}        # polygon description (Ax≦b)
    R::T                    # "cushion" radius
	r_offset::SVector{3,T}
	Q_offset::SMatrix{3,3,T,9}
    function PolygonMRP(A::SMatrix{nh,2,T,nh2},b::SVector{nh,T},R::T) where {nh,nh2,T}
        new{nh,nh2,T}(SA[0,0,0.0],SA[0,0,0.0],A,b,R,SA[0,0,0.0], SA[1 0 0; 0 1 0; 0 0 1.0])
    end
end

mutable struct EllipsoidMRP{T} <: AbstractPrimitiveMRP
	# x'*Q*x ≦ 1.0
	r::SVector{3,T}
    p::SVector{3,T}
    P::SMatrix{3,3,T,9}
	U::SMatrix{3,3,T,9}
	F::Eigen{T, T, SMatrix{3, 3, T, 9}, SVector{3, T}}
	r_offset::SVector{3,T}
	Q_offset::SMatrix{3,3,T,9}
    function Ellipsoid(P::SMatrix{3,3,T,9}) where {T}
        new{T}(
		SA[0,0,0.0],
		SA[0,0,0.0],
		P,
		SMatrix{3,3}(cholesky(P).U),
		eigen(P),
		SA[0,0,0.0],
		SA[1 0 0; 0 1 0; 0 0 1.0]
		)
    end
end
