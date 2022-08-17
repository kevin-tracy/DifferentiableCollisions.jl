@inline function proximity(prim1::Polytope{n1,n1_3,T},prim2::Polytope{n2,n2_3,T}; pdip_tol::Float64 = 1e-6, verbose::Bool = false) where {n1,n1_3,n2, n2_3,T}

    # quaternion specific
    G_ort1, h_ort1, _, _ = problem_matrices(prim1,prim1.r,prim1.q)
    G_ort2, h_ort2, _, _ = problem_matrices(prim2,prim2.r,prim2.q)

    # create and solve SOCP
    G = [G_ort1;G_ort2]
    h = [h_ort1;h_ort2]
    c = SA[0,0,0,1.0]

    x,s,z = solve_lp(c,G,h; pdip_tol = pdip_tol,verbose = verbose)
    return x[4], x[SA[1,2,3]]
end

@inline function proximity_floor(prim1::Polytope{n1,n1_3,T};
                                 pdip_tol::Float64 = 1e-6,
                                 verbose::Bool = false,
                                 basement::Float64 = -10.0) where {n1,n1_3,T}

    # quaternion specific
    G_ort1, h_ort1, _, _ = problem_matrices(prim1,prim1.r,prim1.q)

    # create and solve SOCP, add x[3] <= 0 constraint for the floor
    G = [G_ort1;SA[0 0 1.0 basement]]
    h = [h_ort1;basement]
    c = SA[0,0,0,1.0]

    x,s,z = solve_lp(c,G,h; pdip_tol = pdip_tol,verbose = verbose)
    return x[4], x[SA[1,2,3]]
end

# derivatives for poly poly interaction
@inline function kkt_R(prim1::Polytope{n1,n1_3,T},
                       prim2::Polytope{n2,n2_3,T},
                       x::SVector{nx,T1},
                       s::SVector{nz,T7},
                       z::SVector{nz,T2},
                       r1::SVector{3,T3},
                       q1::SVector{4,T4},
                       r2::SVector{3,T5},
                       q2::SVector{4,T6}) where {nx,nz,T1,T2,T3,T4,T5,T6,T7,n1,n1_3,n2,n2_3,T}

    # quaternion specific
    G_ort1, h_ort1, _, _ = problem_matrices(prim1,r1,q1)
    G_ort2, h_ort2, _, _ = problem_matrices(prim2,r2,q2)

    # create and solve SOCP
    G = [G_ort1;G_ort2]
    h = [h_ort1;h_ort2]
    c = SA[0,0,0,1.0]

    [
    c + G'*z;
    (h - G*x) .* z
    ]
end

@inline function diff_lp(prim1::Polytope{n1,n1_3,T},
                         prim2::Polytope{n2,n2_3,T},
                         x::SVector{nx,T},
                         s::SVector{nz,T},
                         z::SVector{nz,T},
                         G::SMatrix{nz,nx,T,nznx},
                         h::SVector{nz,T}) where {n1,n1_3,n2,n2_3,nx,nz,nznx,T}

    idx_x = SVector{nx}(1:nx)
    idx_z = SVector{nz}((nx + 1):(nx + nz))
    idx_r1 = SVector{3}(1:3)
    idx_q1 = SVector{4}(4:7)
    idx_r2 = SVector{3}(8:10)
    idx_q2 = SVector{4}(11:14)

    # @btime ForwardDiff.jacobian(_θ -> kkt_R($prim1,$prim2,$x,$s,$z,_θ[$idx_r1],_θ[$idx_q1],_θ[$idx_r2],_θ[$idx_q2]), [$prim1.r;$prim1.q;$prim2.r;$prim2.q])

    dR_dθ=ForwardDiff.jacobian(_θ -> kkt_R(prim1,prim2,x,s,z,_θ[idx_r1],_θ[idx_q1],_θ[idx_r2],_θ[idx_q2]), [prim1.r;prim1.q;prim2.r;prim2.q])

    r1 = -dR_dθ[idx_x,:]
    r2 = -dR_dθ[idx_z,:]
    Z = Diagonal(z)
    # S = Diagonal(s)
    S = Diagonal(h - G*x)
    ∂x = (G'*((S\Z)*G))\(r1 - G'*(S\r2))
    ∂x[SA[1,2,3,4],:]
end
@inline function proximity_jacobian(prim1::Polytope{n1,n1_3,T},
                                    prim2::Polytope{n2,n2_3,T};
                                    pdip_tol = 1e-6,
                                    verbose = false) where {n1,n1_3,n2,n2_3,T}
    # quaternion specific
    G_ort1, h_ort1, _, _ = problem_matrices(prim1,prim1.r,prim1.q)
    G_ort2, h_ort2, _, _ = problem_matrices(prim2,prim2.r,prim2.q)

    # create and solve SOCP
    G = [G_ort1;G_ort2]
    h = [h_ort1;h_ort2]
    c = SA[0,0,0,1.0]

    x,s,z = solve_lp(c,G,h; pdip_tol = pdip_tol,verbose = verbose)

    # @btime diff_lp($prim1,$prim2,$x,$s,$z,$G)

    ∂x_∂state = diff_lp(prim1,prim2,x,s,z,G,h)
    x[4], x[SA[1,2,3]], ∂x_∂state
end
