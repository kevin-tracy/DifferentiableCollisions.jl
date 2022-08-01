@inline function mrp_from_q(q::SVector{4,T}) where {T}
    return q[SA[2,3,4]]/(1+q[1])
end

@inline function q_from_mrp(p::SVector{3,T}) where {T}
    return (1/(1+dot(p,p)))*vcat((1-dot(p,p)),2*p)
end

@inline function dcm_from_q(q::SVector{4,T}) where {T}
    q4,q1,q2,q3 = normalize(q)

    # DCM
    Q = @SArray [(2*q1^2+2*q4^2-1)   2*(q1*q2 - q3*q4)   2*(q1*q3 + q2*q4);
          2*(q1*q2 + q3*q4)  (2*q2^2+2*q4^2-1)   2*(q2*q3 - q1*q4);
          2*(q1*q3 - q2*q4)   2*(q2*q3 + q1*q4)  (2*q3^2+2*q4^2-1)]
end

@inline function dcm_from_mrp(p::SVector{3,T}) where {T}
    p1,p2,p3 = p
    den = (p1^2 + p2^2 + p3^2 + 1)^2
    a = (4*p1^2 + 4*p2^2 + 4*p3^2 - 4)
    SA[
    (-((8*p2^2+8*p3^2)/den-1)*den)   (8*p1*p2 + p3*a)     (8*p1*p3 - p2*a);
    (8*p1*p2 - p3*a) (-((8*p1^2 + 8*p3^2)/den - 1)*den)   (8*p2*p3 + p1*a);
    (8*p1*p3 + p2*a)  (8*p2*p3 - p1*a)  (-((8*p1^2 + 8*p2^2)/den - 1)*den)
    ]/den
end

# CAPSULE
@inline function capsule_problem_matrices(R::T,L::T,r::SVector{3,T1},n_Q_b::SMatrix{3,3,T2,9}) where {T,T1,T2}
    bx = n_Q_b*SA[1,0,0.0]
    G_soc_top = SA[0 0 0.0 -R 0]
    G_soc_bot = hcat(-Diagonal(SA[1,1,1.0]), SA[0,0,0.0], bx )
    G_soc = [G_soc_top;G_soc_bot]
    h_soc = [0; -(r)]
    G_ort = SA[0 0 0 -L/2 1; 0 0 0 -L/2 -1.0]
    h_ort = SA[0,0.0]
    G_ort, h_ort, G_soc, h_soc
end
@inline function problem_matrices(capsule::Capsule{T},r::SVector{3,T1},q::SVector{4,T2}) where {T,T1,T2}
    n_Q_b = dcm_from_q(q)
    capsule_problem_matrices(capsule.R,capsule.L,r,n_Q_b)
end
@inline function problem_matrices(capsule::CapsuleMRP{T},r::SVector{3,T1},p::SVector{3,T2}) where {T,T1,T2}
    n_Q_b = dcm_from_mrp(p)
    capsule_problem_matrices(capsule.R,capsule.L,r,n_Q_b)
end

# CONE
@inline function problem_matrices(cone::Cone{T},r::SVector{3,T1},q::SVector{4,T2}) where {T,T1,T2}
    n_Q_b = dcm_from_q(q)
    cone_problem_matrices(cone.H,cone.β,r,n_Q_b)
end
@inline function problem_matrices(cone::ConeMRP{T},r::SVector{3,T1},p::SVector{3,T2}) where {T,T1,T2}
    n_Q_b = dcm_from_mrp(p)
    cone_problem_matrices(cone.H,cone.β,r,n_Q_b)
end
@inline function cone_problem_matrices(H::T,β::T,r::SVector{3,T1},n_Q_b::SMatrix{3,3,T2,9}) where {T,T1,T2}
    tanβ = tan(β)
    E = Diagonal(SA[tanβ,1,1.0])
    bx = n_Q_b*SA[1,0,0]
    EQt = E*n_Q_b'
    h_soc = -EQt*r
    G_soc = [(-EQt) (-SA[tanβ*3*H/4,0,0])]
    G_ort = SA[bx[1] bx[2] bx[3] -H/4]
    h_ort = SA[dot(bx,r)]
    G_ort, h_ort, G_soc, h_soc
end

# @inline function problem_matrices(prim::P) where {P<:AbstractPrimitive}
#     problem_matrices(prim,prim.r,prim.q)
# end
