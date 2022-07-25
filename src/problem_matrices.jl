
@inline function dcm_from_q(q::SVector{4,T}) where {T}
    q4,q1,q2,q3 = normalize(q)

    # DCM
    Q = @SArray [(2*q1^2+2*q4^2-1)   2*(q1*q2 - q3*q4)   2*(q1*q3 + q2*q4);
          2*(q1*q2 + q3*q4)  (2*q2^2+2*q4^2-1)   2*(q2*q3 - q1*q4);
          2*(q1*q3 - q2*q4)   2*(q2*q3 + q1*q4)  (2*q3^2+2*q4^2-1)]
end


# solve for collision
@inline function problem_matrices(capsule::Capsule{T},r::SVector{3,T1},q::SVector{4,T2}) where {T,T1,T2}
    n_Q_b = dcm_from_q(q)
    bx = n_Q_b*SA[1,0,0.0]
    G_soc_top = SA[0 0 0.0 -capsule.R 0]
    G_soc_bot = hcat(-Diagonal(SA[1,1,1.0]), SA[0,0,0.0], bx )
    G_soc = [G_soc_top;G_soc_bot]
    h_soc = [0; -(r)]
    G_ort = SA[0 0 0 -capsule.L/2 1; 0 0 0 -capsule.L/2 -1.0]
    h_ort = SA[0,0.0]
    G_ort, h_ort, G_soc, h_soc
end

@inline function problem_matrices(cone::Cone{T},r::SVector{3,T1},q::SVector{4,T2}) where {T,T1,T2}
    # TODO: remove last column
    E = Diagonal(SA[tan(cone.β),1,1.0])
    n_Q_b = dcm_from_q(q)
    bx = n_Q_b*SA[1,0,0]
    EQt = E*n_Q_b'
    h_soc = -EQt*r
    G_soc = [(-EQt) (-SA[tan(cone.β)*cone.H/2,0,0])]
    # G_ort = hcat(bx', -cone.H/2)
    G_ort = SA[bx[1] bx[2] bx[3] -cone.H/2]
    h_ort = SA[dot(bx,r)]
    G_ort, h_ort, G_soc, h_soc
end

@inline function problem_matrices(prim::P) where {P<:AbstractPrimitive}
    problem_matrices(prim,prim.r,prim.q)
end
