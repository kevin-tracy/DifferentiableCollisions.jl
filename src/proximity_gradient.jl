@inline function lag_con_part(capsule::P1,
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
    _,G,h,_,_,_ = combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1,G_ort2, h_ort2, G_soc2, h_soc2)

    return z'*(G*x - h)
end
@inline function lag_con_part(capsule::P1,
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
    c,G,h,_,_,_ = combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1,G_ort2, h_ort2, G_soc2, h_soc2)

    return z'*(G*x - h)
end
@inline function obj_val_grad(capsule::P1,
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

   ForwardDiff.gradient(_θ -> lag_con_part(capsule,cone,x,s,z,_θ[idx_r1],_θ[idx_q1],_θ[idx_r2],_θ[idx_q2],idx_ort,idx_soc1,idx_soc2), [capsule.r;capsule.q;cone.r;cone.q])
end
@inline function obj_val_grad(capsule::P1,
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

   ForwardDiff.gradient(_θ -> lag_con_part(capsule,cone,x,s,z,_θ[idx_r1],_θ[idx_p1],_θ[idx_r2],_θ[idx_p2],idx_ort,idx_soc1,idx_soc2), [capsule.r;capsule.p;cone.r;cone.p])
end

@inline function proximity_gradient(prim1::P1,prim2::P2; pdip_tol::Float64 = 1e-6, verbose = false) where {P1 <: AbstractPrimitive, P2 <: AbstractPrimitive}

    # quaternion specific
    G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(prim1,prim1.r,prim1.q)
    G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(prim2,prim2.r,prim2.q)

    # create and solve SOCP
    c,G,h,idx_ort,idx_soc1,idx_soc2 = combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1,G_ort2, h_ort2, G_soc2, h_soc2)
    x,s,z = solve_socp(c,G,h,idx_ort,idx_soc1,idx_soc2; verbose = verbose, pdip_tol = pdip_tol)

    α = x[4]

    ∂α_∂state= obj_val_grad(prim1,prim2,x,s,z,idx_ort,idx_soc1,idx_soc2)

    return α, ∂α_∂state
end
@inline function proximity_gradient(prim1::P1,prim2::P2; pdip_tol::Float64 = 1e-6, verbose = false) where {P1 <: AbstractPrimitiveMRP, P2 <: AbstractPrimitiveMRP}

    # MRP specific
    G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(prim1,prim1.r,prim1.p)
    G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(prim2,prim2.r,prim2.p)

    # create and solve SOCP
    c,G,h,idx_ort,idx_soc1,idx_soc2 = combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1,G_ort2, h_ort2, G_soc2, h_soc2)
    x,s,z = solve_socp(c,G,h,idx_ort,idx_soc1,idx_soc2; verbose = verbose, pdip_tol = pdip_tol)

    α = x[4]

    ∂α_∂state= obj_val_grad(prim1,prim2,x,s,z,idx_ort,idx_soc1,idx_soc2)

    return α, ∂α_∂state
end
