

function proximity(prim1::P1,prim2::P2; pdip_tol::Float64 = 1e-6) where {P1 <: AbstractPrimitive, P2 <: AbstractPrimitive}

    # quaternion specific (for now TODO: add MRP support)
    G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(prim1,prim1.r,prim1.q)
    G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(prim2,prim2.r,prim2.q)

    # create and solve SOCP
    c,G,h,idx_ort,idx_soc1,idx_soc2 = combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1,G_ort2, h_ort2, G_soc2, h_soc2)
    x,s,z = solve_socp(c,G,h,idx_ort,idx_soc1,idx_soc2; verbose = false, pdip_tol = pdip_tol)

    return x[4], x[SA[1,2,3]]
end
@inline function kkt_R(capsule::P1,
           cone::P2,
           x::SVector{nx,T1},
           s::SVector{nz,T7},
           z::SVector{nz,T2},
           r1::SVector{3,T3},
           q1::SVector{4,T4},
           r2::SVector{3,T5},
           q2::SVector{4,T6},
           idx_ort::SVector{n_ort,Ti},
           idx_soc1::SVector{n_soc1,Ti},
           idx_soc2::SVector{n_soc2,Ti}) where {nx,nz,n_ort,n_soc1,n_soc2,Ti,T1,T2,T3,T4,T5,T6,T7,P1 <: AbstractPrimitive, P2 <: AbstractPrimitive}

    G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(capsule,r1,q1)
    G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(cone,r2,q2)
    c,G_,h_,_,_,_ = combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1,G_ort2, h_ort2, G_soc2, h_soc2)
    [
    c + G_'*z;
    cone_product(h_ - G_*x, z, idx_ort, idx_soc1, idx_soc2)
    ]
end
@inline function diff_socp(capsule::P1,
           cone::P2,
           x::SVector{nx,T1},
           s::SVector{nz,T7},
           z::SVector{nz,T2},
           idx_ort::SVector{n_ort,Ti},
           idx_soc1::SVector{n_soc1,Ti},
           idx_soc2::SVector{n_soc2,Ti}) where {nx,nz,n_ort,n_soc1,n_soc2,Ti,T1,T2,T7,P1 <: AbstractPrimitive, P2 <: AbstractPrimitive}


   idx_x = SVector{nx}(1:nx)
   idx_z = SVector{nz}((nx + 1):(nx + nz))
   idx_r1 = SVector{3}(1:3)
   idx_q1 = SVector{4}(4:7)
   idx_r2 = SVector{3}(8:10)
   idx_q2 = SVector{4}(11:14)

   dR_dθ=ForwardDiff.jacobian(_θ -> kkt_R(capsule,cone,x,s,z,_θ[idx_r1],_θ[idx_q1],_θ[idx_r2],_θ[idx_q2],idx_ort,idx_soc1,idx_soc2), [capsule.r;capsule.q;cone.r;cone.q])

   G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(capsule,capsule.r,capsule.q)
   G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(cone,cone.r,cone.q)

   _,G,_,_,_,_ = combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1,G_ort2, h_ort2, G_soc2, h_soc2)

   r1 = -dR_dθ[idx_x,:]
   r2 = -dR_dθ[idx_z,:]
   Z = scaling_2(z[idx_ort],arrow(z[idx_soc1]),arrow(z[idx_soc2]))
   S = NT_scaling_2(s[idx_ort],arrow(s[idx_soc1]), cholesky(arrow(s[idx_soc1])), arrow(s[idx_soc2]), cholesky(arrow(s[idx_soc2])))
   ∂x = (G'*((S\Z)*G))\(r1 - G'*(S\r2))
   vec(∂x[4,:])
end

function proximity_gradient(prim1::P1,prim2::P2; pdip_tol::Float64 = 1e-6) where {P1 <: AbstractPrimitive, P2 <: AbstractPrimitive}

    # quaternion specific (for now TODO: add MRP support)
    G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(prim1,prim1.r,prim1.q)
    G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(prim2,prim2.r,prim2.q)

    # create and solve SOCP
    c,G,h,idx_ort,idx_soc1,idx_soc2 = combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1,G_ort2, h_ort2, G_soc2, h_soc2)
    x,s,z = solve_socp(c,G,h,idx_ort,idx_soc1,idx_soc2; verbose = false, pdip_tol = pdip_tol)

    ∂α_∂state= diff_socp(prim1,prim2,x,s,z,idx_ort,idx_soc1,idx_soc2)

    return x[4], x[SA[1,2,3]], ∂α_∂state
end
