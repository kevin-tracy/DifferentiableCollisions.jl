import Pkg; Pkg.activate("/Users/kevintracy/.julia/dev/DCD/extras")

using LinearAlgebra
using Convex, ECOS
using JLD2
using StaticArrays
import DCD
import MeshCat as mc

@load "/Users/kevintracy/.julia/dev/DCD/extras/polytopes.jld2"

A1 = SMatrix{14,3}(A1)
b1 = SVector{14}(b1)
A2 = SMatrix{8,3}(A2)
b2 = SVector{8}(b2)

P1 = DCD.Polytope(A1,b1)
P2 = DCD.Polytope(A2,b2)

P1.r = 1*(@SVector randn(3))
P1.q = normalize((@SVector randn(4)))
P2.r = 1*(@SVector randn(3))
P2.q = normalize((@SVector randn(4)))


x = Variable(3)
α = Variable()

Q1 = DCD.dcm_from_q(P1.q)
Q2 = DCD.dcm_from_q(P2.q)

prob = minimize(α)
prob.constraints += α >= 0
prob.constraints += P1.A*Q1'*(x - P1.r) <= P1.b*α
prob.constraints += P2.A*Q2'*(x - P2.r) <= P2.b*α

solve!(prob, ECOS.Optimizer)

α = α.value
x = vec(x.value)

# my QP stuff
@inline function polytope_problem_matrices(A::SMatrix{nh,3,T1,nh3},b::SVector{nh,T2},r::SVector{3,T3},n_Q_b::SMatrix{3,3,T4,9}) where {nh,nh3,T1,T2,T3,T4}
    AQt = A*n_Q_b'
    G_ort = [AQt  -b]
    h_ort = AQt*r
    G_ort, h_ort, nothing, nothing
end
@inline function problem_matrices(polytope::DCD.Polytope{n,n3,T},r::SVector{3,T1},q::SVector{4,T2}) where {n,n3,T,T1,T2}
    n_Q_b = DCD.dcm_from_q(q)
    polytope_problem_matrices(polytope.A,polytope.b,r,n_Q_b)
end


G1,h1, _, _ = problem_matrices(P1,P1.r,P1.q)
G2,h2, _, _ = problem_matrices(P2,P2.r,P2.q)
#
G = [G1;G2]
h = [h1;h2]
#
include("/Users/kevintracy/.julia/dev/DCD/src/lp_solver.jl")
#
# z = Variable(4)
# prob = minimize(z[4], G*z <= h)
# solve!(prob, ECOS.Optimizer)
xs,zs,ss = pdip(SA[0,0,0,1.0],G,h; tol = 1e-6, verbose = true)
#
#
# @btime pdip(SA[0,0,0,1.0],$G,$h; tol = 1e-6)
#
# @btime pdip_init(SA[0,0,0,1.0],$G,$h)

# @btime problem_matrices($P1,$P1.r,$P1.q)
# plotting
# vis = mc.Visualizer()
# open(vis)

c1 = [245, 155, 66]/255
c2 = [2,190,207]/255


if α > 1
    trans_1 = 1.0
    trans_2 = 0.5
else
    trans_2 = 1.0
    trans_1 = 0.5
end

DCD.build_primitive!(vis, P1, :polytope1; α = 1.0,color = mc.RGBA(c1..., trans_1))
DCD.build_primitive!(vis, P2, :polytope2; α = 1.0,color = mc.RGBA(c2..., trans_1))
DCD.build_primitive!(vis, P1, :polytope1_big; α = α,color = mc.RGBA(c1..., trans_2))
DCD.build_primitive!(vis, P2, :polytope2_big; α = α,color = mc.RGBA(c2..., trans_2))

DCD.update_pose!(vis[:polytope1], P1)
DCD.update_pose!(vis[:polytope2], P2)
DCD.update_pose!(vis[:polytope1_big], P1)
DCD.update_pose!(vis[:polytope2_big], P2)

sph_p1 = mc.HyperSphere(mc.Point(x...), 0.1)
mc.setobject!(vis[:p1], sph_p1, mc.MeshPhongMaterial(color = mc.RGBA(1.0,0,0,1.0)))

@info α
@test norm(x - xs[1:3],1) < 1e-4
@test abs(α - xs[4]) < 1e-4
