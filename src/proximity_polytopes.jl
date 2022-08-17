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

@inline function proximity_floor(prim1::Polytope{n1,n1_3,T}; pdip_tol::Float64 = 1e-6, verbose::Bool = false) where {n1,n1_3,T}

    # quaternion specific
    G_ort1, h_ort1, _, _ = problem_matrices(prim1,prim1.r,prim1.q)

    # create and solve SOCP, add x[3] <= 0 constraint for the floor
    basement = -10.0
    G = [G_ort1;SA[0 0 1.0 basement]]
    h = [h_ort1;basement]
    c = SA[0,0,0,1.0]

    x,s,z = solve_lp(c,G,h; pdip_tol = pdip_tol,verbose = verbose)
    return x[4], x[SA[1,2,3]]
end
