

@inline function proximity(prim1::P1,prim2::P2; pdip_tol::Float64 = 1e-6, verbose::Bool = false) where {P1 <: AbstractPrimitive, P2 <: AbstractPrimitive}

    # quaternion specific
    G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(prim1,prim1.r,prim1.q)
    G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(prim2,prim2.r,prim2.q)

    # create and solve SOCP
    c,G,h,idx_ort,idx_soc1,idx_soc2 = combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1,G_ort2, h_ort2, G_soc2, h_soc2)
    x,s,z = solve_socp(c,G,h,idx_ort,idx_soc1,idx_soc2; verbose = verbose, pdip_tol = pdip_tol)

    return x[4], x[SA[1,2,3]]
end
@inline function proximity(prim1::P1,prim2::P2; pdip_tol::Float64 = 1e-6, verbose::Bool = false) where {P1 <: AbstractPrimitiveMRP, P2 <: AbstractPrimitiveMRP}

    # MRP specific
    G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(prim1,prim1.r,prim1.p)
    G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(prim2,prim2.r,prim2.p)

    # create and solve SOCP
    c,G,h,idx_ort,idx_soc1,idx_soc2 = combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1,G_ort2, h_ort2, G_soc2, h_soc2)
    x,s,z = solve_socp(c,G,h,idx_ort,idx_soc1,idx_soc2; verbose = verbose, pdip_tol = pdip_tol)

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

    # quaternion specific problem matrices
    G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(capsule,r1,q1)
    G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(cone,r2,q2)
    c,G_,h_,_,_,_ = combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1,G_ort2, h_ort2, G_soc2, h_soc2)
    [
    c + G_'*z;
    cone_product(h_ - G_*x, z, idx_ort, idx_soc1, idx_soc2)
    ]
end
@inline function kkt_R(capsule::P1,
           cone::P2,
           x::SVector{nx,T1},
           s::SVector{nz,T7},
           z::SVector{nz,T2},
           r1::SVector{3,T3},
           p1::SVector{3,T4},
           r2::SVector{3,T5},
           p2::SVector{3,T6},
           idx_ort::SVector{n_ort,Ti},
           idx_soc1::SVector{n_soc1,Ti},
           idx_soc2::SVector{n_soc2,Ti}) where {nx,nz,n_ort,n_soc1,n_soc2,Ti,T1,T2,T3,T4,T5,T6,T7,P1 <: AbstractPrimitiveMRP, P2 <: AbstractPrimitiveMRP}

    # MRP specific problem matrices
    G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(capsule,r1,p1)
    G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(cone,r2,p2)
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

   # quaternion specific problem matrices
   G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(capsule,capsule.r,capsule.q)
   G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(cone,cone.r,cone.q)

   _,G,_,_,_,_ = combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1,G_ort2, h_ort2, G_soc2, h_soc2)

   r1 = -dR_dθ[idx_x,:]
   r2 = -dR_dθ[idx_z,:]
   Z = scaling_2(z[idx_ort],arrow(z[idx_soc1]),arrow(z[idx_soc2]))
   S = NT_scaling_2(s[idx_ort],arrow(s[idx_soc1]), cholesky(arrow(s[idx_soc1])), arrow(s[idx_soc2]), cholesky(arrow(s[idx_soc2])))
   ∂x = (G'*((S\Z)*G))\(r1 - G'*(S\r2))
   ∂x[SA[1,2,3,4],:]
end
@inline function diff_socp(capsule::P1,
           cone::P2,
           x::SVector{nx,T1},
           s::SVector{nz,T7},
           z::SVector{nz,T2},
           idx_ort::SVector{n_ort,Ti},
           idx_soc1::SVector{n_soc1,Ti},
           idx_soc2::SVector{n_soc2,Ti}) where {nx,nz,n_ort,n_soc1,n_soc2,Ti,T1,T2,T7,P1 <: AbstractPrimitiveMRP, P2 <: AbstractPrimitiveMRP}


   idx_x = SVector{nx}(1:nx)
   idx_z = SVector{nz}((nx + 1):(nx + nz))
   idx_r1 = SVector{3}(1:3)
   idx_p1 = SVector{3}(4:6)
   idx_r2 = SVector{3}(7:9)
   idx_p2 = SVector{3}(10:12)

   dR_dθ=ForwardDiff.jacobian(_θ -> kkt_R(capsule,cone,x,s,z,_θ[idx_r1],_θ[idx_p1],_θ[idx_r2],_θ[idx_p2],idx_ort,idx_soc1,idx_soc2), [capsule.r;capsule.p;cone.r;cone.p])

   # MRP specific
   G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(capsule,capsule.r,capsule.p)
   G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(cone,cone.r,cone.p)

   _,G,_,_,_,_ = combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1,G_ort2, h_ort2, G_soc2, h_soc2)

   r1 = -dR_dθ[idx_x,:]
   r2 = -dR_dθ[idx_z,:]
   Z = scaling_2(z[idx_ort],arrow(z[idx_soc1]),arrow(z[idx_soc2]))
   S = NT_scaling_2(s[idx_ort],arrow(s[idx_soc1]), cholesky(arrow(s[idx_soc1])), arrow(s[idx_soc2]), cholesky(arrow(s[idx_soc2])))
   ∂x = (G'*((S\Z)*G))\(r1 - G'*(S\r2))
   ∂x[SA[1,2,3,4],:]
end
@inline function diff_socp_slow(capsule::P1,
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

   # TODO: MRP SUPPORT
   G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(capsule,capsule.r,capsule.q)
   G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(cone,cone.r,cone.q)

   _,G,h,_,_,_ = combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1,G_ort2, h_ort2, G_soc2, h_soc2)

   # Z = Matrix(blockdiag(sparse(Diagonal(z[idx_ort])),sparse(DCD.arrow(z[idx_soc1])),sparse(DCD.arrow(z[idx_soc2]))))
   Z = block_diag(Diagonal(z[idx_ort]),block_diag(arrow(z[idx_soc1]), arrow(z[idx_soc2])))
   s̃ = h - G*x
   S = block_diag(Diagonal(s̃[idx_ort]),block_diag(arrow(s̃[idx_soc1]), arrow(s̃[idx_soc2])))
   dR_dw = [zeros(length(x),length(x)) G'; -Z*G S]
   dw_dθ = -dR_dw\dR_dθ
   dw_dθ[SA[1,2,3,4],:]
end
@inline function proximity_jacobian(prim1::P1,prim2::P2; pdip_tol::Float64 = 1e-6, verbose = false) where {P1 <: AbstractPrimitive, P2 <: AbstractPrimitive}

    # quaternion specific
    G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(prim1,prim1.r,prim1.q)
    G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(prim2,prim2.r,prim2.q)

    # create and solve SOCP
    c,G,h,idx_ort,idx_soc1,idx_soc2 = combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1,G_ort2, h_ort2, G_soc2, h_soc2)
    x,s,z = solve_socp(c,G,h,idx_ort,idx_soc1,idx_soc2; verbose = verbose, pdip_tol = pdip_tol)

    ∂x_∂state= diff_socp(prim1,prim2,x,s,z,idx_ort,idx_soc1,idx_soc2)

    return x[4], x[SA[1,2,3]], ∂x_∂state
end
@inline function proximity_jacobian(prim1::P1,prim2::P2; pdip_tol::Float64 = 1e-6, verbose = false) where {P1 <: AbstractPrimitiveMRP, P2 <: AbstractPrimitiveMRP}

    # MRP specific
    G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(prim1,prim1.r,prim1.p)
    G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(prim2,prim2.r,prim2.p)

    # create and solve SOCP
    c,G,h,idx_ort,idx_soc1,idx_soc2 = combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1,G_ort2, h_ort2, G_soc2, h_soc2)
    x,s,z = solve_socp(c,G,h,idx_ort,idx_soc1,idx_soc2; verbose = verbose, pdip_tol = pdip_tol)

    ∂x_∂state= diff_socp(prim1,prim2,x,s,z,idx_ort,idx_soc1,idx_soc2)

    return x[4], x[SA[1,2,3]], ∂x_∂state
end
@inline function proximity_jacobian_slow(prim1::P1,prim2::P2; pdip_tol::Float64 = 1e-6) where {P1 <: AbstractPrimitive, P2 <: AbstractPrimitive}

    # quaternion specific (for now TODO: add MRP support)
    G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(prim1,prim1.r,prim1.q)
    G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(prim2,prim2.r,prim2.q)

    # create and solve SOCP
    c,G,h,idx_ort,idx_soc1,idx_soc2 = combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1,G_ort2, h_ort2, G_soc2, h_soc2)
    x,s,z = solve_socp(c,G,h,idx_ort,idx_soc1,idx_soc2; verbose = false, pdip_tol = pdip_tol)

    ∂x_∂state= diff_socp_slow(prim1,prim2,x,s,z,idx_ort,idx_soc1,idx_soc2)

    return x[4], x[SA[1,2,3]], ∂x_∂state
end
