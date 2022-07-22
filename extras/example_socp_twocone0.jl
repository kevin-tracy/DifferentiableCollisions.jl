cd("/Users/kevintracy/.julia/dev/DCD/extras")
import Pkg; Pkg.activate(".")
using DCD
import ForwardDiff as FD
include("primitives.jl")
using LinearAlgebra, StaticArrays, Convex, ECOS
function dcm_from_q(q::SVector{4,T}) where {T}
    q4,q1,q2,q3 = normalize(q)

    # DCM
    Q = @SArray [(2*q1^2+2*q4^2-1)   2*(q1*q2 - q3*q4)   2*(q1*q3 + q2*q4);
          2*(q1*q2 + q3*q4)  (2*q2^2+2*q4^2-1)   2*(q2*q3 - q1*q4);
          2*(q1*q3 - q2*q4)   2*(q2*q3 + q1*q4)  (2*q3^2+2*q4^2-1)]
end


# solve for collision
function problem_matrices(capsule::Capsule{T}) where {T}
    n_Q_b = dcm_from_q(capsule.q)
    bx = n_Q_b*SA[1,0,0.0]
    G_soc_top = SA[0 0 0.0 -capsule.R 0]
    G_soc_bot = hcat(-Diagonal(SA[1,1,1.0]), SA[0,0,0.0], bx )
    G_soc = [G_soc_top;G_soc_bot]
    h_soc = [0; -(capsule.r)]
    G_ort = SA[0 0 0 -capsule.L/2 1; 0 0 0 -capsule.L/2 -1]
    h_ort = SA[0,0]
    G_ort, h_ort, G_soc, h_soc
end

function problem_matrices(cone::Cone{T}) where {T}
    # TODO: remove last column
    E = Diagonal(SA[tan(cone.β),1,1.0])
    n_Q_b = dcm_from_q(cone.q)
    bx = n_Q_b*SA[1,0,0]
    EQt = E*n_Q_b'
    h_soc = -EQt*cone.r
    G_soc = [(-EQt) (-SA[tan(cone.β)*cone.H/2,0,0])]
    G_ort = hcat(bx', -cone.H/2)
    h_ort = SA[dot(bx,cone.r)]
    G_ort, h_ort, G_soc, h_soc
end

function problem_matrices(prim::P,r::SVector{3,T},q::SVector{4,T}) where {P<:AbstractPrimitive,T}
    prim.r = r
    prim.q = q
    problem_matrices(prim)
end
# @inline function cone_product(s::SVector{n,T},z::SVector{n,T},idx_ort::SVector{n_ort,Ti}, idx_soc1::SVector{n_soc1, Ti},idx_soc2) where {n,T,n_ort,n_soc1,n_soc2,Ti}
#     idx_ort = SVector{n_ort}(1:n_ort)
#     idx_soc1 = SVector{n_soc1}((n_ort + 1):(n_ort + n_soc1))
#     idx_soc2 = SVector{n_soc2}((n_ort + n_soc1 + 1):(n_ort + n_soc1 + n_soc2))
#
#     s_ort = s[idx_ort]
#     z_ort = z[idx_ort]
#     s_soc1 = s[idx_soc1]
#     z_soc1 = z[idx_soc1]
#     s_soc2 = s[idx_soc2]
#     z_soc2 = z[idx_soc2]
#
#     [s_ort .* z_ort; soc_cone_product(s_soc1,z_soc1);soc_cone_product(s_soc2,z_soc2)]
# end
function R(capsule::Capsule{T},
           cone::Cone{T},
           x,
           z,
           r1,
           q1,
           r2,
           q2,
           idx_ort::SVector{n_ort,Ti},
           idx_soc1::SVector{n_soc1,Ti},
           idx_soc2::SVector{n_soc2,Ti}) where {T,nx,nz,n_ort,n_soc1,n_soc2,Ti}

    G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(capsule,r1,q1)
    G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(cone,r2,q2)

    n_ort1 = length(h_ort1)
    n_ort2 = length(h_ort2)
    # n_soc1 = length(h_soc1)
    # n_soc2 = length(h_soc2)
    # n_ort = n_ort1 + n_ort2

    G_ort_top = G_ort1
    G_ort_bot = hcat(G_ort2, (@SVector zeros(n_ort2))) # add a column for γ (capsule)

    G_soc_top = G_soc1
    G_soc_bot = hcat(G_soc2, (@SVector zeros(n_soc2))) # add a column for γ (capsule)

    G_ = [G_ort_top;G_ort_bot;G_soc_top;G_soc_bot]
    h_ = [h_ort1;h_ort2;h_soc1;h_soc2]


    c = SA[0,0,0,1.0,0]

    [
    c + G_'*z;
    DCD.cone_product(h_ - G_*x, z, idx_ort, idx_soc1, idx_soc2)
    ]
end

function solve_alpha(capsule::Capsule{T},
           cone::Cone{T},
           r1,
           q1,
           r2,
           q2,
           idx_ort::SVector{n_ort,Ti},
           idx_soc1::SVector{n_soc1,Ti},
           idx_soc2::SVector{n_soc2,Ti}) where {T,n_ort,n_soc1,n_soc2,Ti}

    G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(capsule,r1,q1)
    G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(cone,r2,q2)

    n_ort1 = length(h_ort1)
    n_ort2 = length(h_ort2)
    # n_soc1 = length(h_soc1)
    # n_soc2 = length(h_soc2)
    # n_ort = n_ort1 + n_ort2

    G_ort_top = G_ort1
    G_ort_bot = hcat(G_ort2, (@SVector zeros(n_ort2))) # add a column for γ (capsule)

    G_soc_top = G_soc1
    G_soc_bot = hcat(G_soc2, (@SVector zeros(n_soc2))) # add a column for γ (capsule)

    G_ = [G_ort_top;G_ort_bot;G_soc_top;G_soc_bot]
    h_ = [h_ort1;h_ort2;h_soc1;h_soc2]


    x,s,z = DCD.solve_socp(SA[0,0,0,1.0,0],G_,h_,idx_ort,idx_soc1,idx_soc2; verbose = true, pdip_tol = 1e-12)
    [x[4]]
end

# collision between cone and capsule
# vis1 = mc.Visualizer()
# open(vis1)
# vis2 = mc.Visualizer()
# open(vis2)
let
    cone = Cone(2.0,deg2rad(22))
    cone.r = 0.3*(@SVector randn(3))
    cone.q = normalize((@SVector randn(4)))
    build_primitive!(vis1, cone, :cone; α = 1.0,color = mc.RGBA(0.1, 0.7, 0.7, 0.7))
    update_pose!(vis1[:cone],cone)

    capsule = Capsule(.3,1.2)
    capsule.r = (@SVector randn(3))
    capsule.q = normalize((@SVector randn(4)))
    build_primitive!(vis1, capsule, :capsule; α = 1.0,color = mc.RGBA(0.7, 1.0, 0.7, 0.7))
    update_pose!(vis1[:capsule],capsule)
    G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(capsule)
    G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(cone)

    n_ort1_ = length(h_ort1)
    n_ort2_ = length(h_ort2)
    n_soc1_ = length(h_soc1)
    n_soc2_ = length(h_soc2)
    n_ort_ = n_ort1_ + n_ort2_

    G_ort_top = G_ort1
    G_ort_bot = hcat(G_ort2, (@SVector zeros(n_ort2_))) # add a column for γ (capsule)

    G_soc_top = G_soc1
    G_soc_bot = hcat(G_soc2, (@SVector zeros(n_soc2_)))

    G_ = [G_ort_top;G_ort_bot;G_soc_top;G_soc_bot]
    h_ = [h_ort1;h_ort2;h_soc1;h_soc2]

    idx_ort = SVector{n_ort_}(1:n_ort_)
    idx_soc1 = SVector{n_soc1_}((n_ort_ + 1):(n_ort_ + n_soc1_))
    idx_soc2 = SVector{n_soc2_}((n_ort_ + n_soc1_ + 1):(n_ort_ + n_soc1_ + n_soc2_))

    x,s,z = DCD.solve_socp(SA[0,0,0,1.0,0],G_,h_,idx_ort,idx_soc1,idx_soc2; verbose = true, pdip_tol = 1e-12)
    @info "solved"
    ree= DCD.cone_product(h_ - G_*x,z,idx_ort,idx_soc1,idx_soc2)
    @show ree
    @show norm(ree)
    # x = x̄[1:3]
    α = x[4]
    # γ = x̄[5]

    # build big ones
    build_primitive!(vis2, cone, :cone_int; α = α,color = mc.RGBA(0.1, 0.7, 0.7, 0.4))
    update_pose!(vis2[:cone_int],cone)
    build_primitive!(vis2, capsule, :capsule_int; α = α,color = mc.RGBA(0.7, 1.0, 0.7, 0.4))
    update_pose!(vis2[:capsule_int],capsule)

    spha = mc.HyperSphere(mc.Point(x[1:3]...), 0.02)
    mc.setobject!(vis2[:intersec], spha, mc.MeshPhongMaterial(color=mc.RGBA(1.0,0,0,1.0)))


    res =  R(capsule,cone,x,z,capsule.r,capsule.q,cone.r,cone.q,idx_ort,idx_soc1,idx_soc2)

    nx = length(x); nz = length(z)
    idx_x = SVector{nx}(1:length(x))
    idx_z = SVector{nz}((length(x) + 1):(length(x) + length(z)))
    dR_dw=FiniteDiff.finite_difference_jacobian(_w -> R(capsule,cone,_w[idx_x],_w[idx_z],capsule.r,capsule.q,cone.r,cone.q,idx_ort,idx_soc1,idx_soc2),[x;z])
    idx_r1 = SVector{3}(1:3)
    idx_q1 = SVector{4}(4:7)
    idx_r2 = SVector{3}(8:10)
    idx_q2 = SVector{4}(11:14)
    dR_dθ=FiniteDiff.finite_difference_jacobian(_θ -> R(capsule,cone,x,z,_θ[idx_r1],_θ[idx_q1],_θ[idx_r2],_θ[idx_q2],idx_ort,idx_soc1,idx_soc2), [capsule.r;capsule.q;cone.r;cone.q])

    dw_dθ = -dR_dw\dR_dθ

    @show dw_dθ[4,:]

    @show solve_alpha(capsule,cone,capsule.r,capsule.q,cone.r,cone.q,idx_ort,idx_soc1,idx_soc2)
    dα_dθ=FiniteDiff.finite_difference_jacobian(_θ -> solve_alpha(capsule,cone,_θ[idx_r1],_θ[idx_q1],_θ[idx_r2],_θ[idx_q2],idx_ort,idx_soc1,idx_soc2), [capsule.r;capsule.q;cone.r;cone.q])

    @show dα_dθ
    @show dw_dθ[4,:]

    @show norm(vec(dα_dθ) - vec(dw_dθ[4,:]))



end
