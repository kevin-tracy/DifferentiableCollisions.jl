import Pkg; Pkg.activate("/Users/kevintracy/.julia/dev/DCD/extras")

using LinearAlgebra
using Convex, ECOS
using JLD2
using StaticArrays
import DCD
import MeshCat as mc
import ForwardDiff
import FiniteDiff
import ForwardDiff

@load "/Users/kevintracy/.julia/dev/DCD/extras/polytopes.jld2"

A1 = SMatrix{14,3}(A1)
b1 = SVector{14}(b1)
A2 = SMatrix{8,3}(A2)
b2 = SVector{8}(b2)

P1 = DCD.Polytope(A1,b1)
P2 = DCD.Polytope(A2,b2)

# P1.r = 1*(@SVector randn(3))
# P1.q = normalize((@SVector randn(4)))
# P2.r = 1*(@SVector randn(3))
# P2.q = normalize((@SVector randn(4)))
P1.r = SA[1.2096647241865888, 0.22365789474303704, -1.107879218065021]
P1.q = SA[0.6635556477310341, 0.28323132121872296, 0.4804390425297425, 0.4986504261083594]
P2.r = SA[0.1090413455105476, 1.4301005488814234, -0.7007693939321329]
P2.q = SA[-0.48284162532008096, -0.22909387568905024, 0.5391245278225849, 0.6509414135539948]

@inline function kkt_R(prim1::DCD.Polytope{n1,n1_3,T},
                       prim2::DCD.Polytope{n2,n2_3,T},
                       x::SVector{nx,T1},
                       s::SVector{nz,T7},
                       z::SVector{nz,T2},
                       r1::SVector{3,T3},
                       q1::SVector{4,T4},
                       r2::SVector{3,T5},
                       q2::SVector{4,T6}) where {nx,nz,T1,T2,T3,T4,T5,T6,T7,n1,n1_3,n2,n2_3,T}

    # quaternion specific
    G_ort1, h_ort1, _, _ = DCD.problem_matrices(prim1,r1,q1)
    G_ort2, h_ort2, _, _ = DCD.problem_matrices(prim2,r2,q2)

    # create and solve SOCP
    G = [G_ort1;G_ort2]
    h = [h_ort1;h_ort2]
    c = SA[0,0,0,1.0]

    [
    c + G'*z;
    (h - G*x) .* z
    ]
end

@inline function diff_lp(prim1::DCD.Polytope{n1,n1_3,T},
                         prim2::DCD.Polytope{n2,n2_3,T},
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
    S = Diagonal(G*x-h) # NOTE: this could easily be G*x - h
    ∂x = (G'*((S\Z)*G))\(r1 - G'*(S\r2))
    ∂x[SA[1,2,3,4],:]
end
@inline function prox_jac(prim1::DCD.Polytope{n1,n1_3,T},
                          prim2::DCD.Polytope{n2,n2_3,T};
                          pdip_tol = 1e-6,
                          verbose = false) where {n1,n1_3,n2,n2_3,T}
    # quaternion specific
    G_ort1, h_ort1, _, _ = DCD.problem_matrices(prim1,prim1.r,prim1.q)
    G_ort2, h_ort2, _, _ = DCD.problem_matrices(prim2,prim2.r,prim2.q)

    # create and solve SOCP
    G = [G_ort1;G_ort2]
    h = [h_ort1;h_ort2]
    c = SA[0,0,0,1.0]

    x,s,z = DCD.solve_lp(c,G,h; pdip_tol = pdip_tol,verbose = verbose)

    # @btime diff_lp($prim1,$prim2,$x,$s,$z,$G)

    diff_lp(prim1,prim2,x,s,z,G,h)
end

function fd_ver(prim1,prim2,z)
    idx_r1 = SVector{3}(1:3)
    idx_q1 = SVector{4}(4:7)
    idx_r2 = SVector{3}(8:10)
    idx_q2 = SVector{4}(11:14)
    prim1.r = z[idx_r1];
    prim1.q = z[idx_q1];
    prim2.r = z[idx_r2];
    prim2.q = z[idx_q2];
    α_star, x_star = DCD.proximity(prim1,prim2)
    [x_star;α_star]
end



α_star, x_star = DCD.proximity(P1,P2)
J = prox_jac(P1,P2; verbose = true, pdip_tol = 1e-6)
# @btime prox_jac($P1,$P2,pdip_tol = 1e-4)


J = prox_jac(P1,P2; verbose = true, pdip_tol = 1e-6)
J2 = FiniteDiff.finite_difference_jacobian(_z -> fd_ver(P1,P2,_z), [P1.r;P1.q;P2.r;P2.q])
#
@show norm(J-J2,Inf)

_,_,J3 = DCD.proximity_jacobian(P1,P2)
