import Pkg; Pkg.activate("/Users/kevintracy/.julia/dev/DCD/extras")

using LinearAlgebra
using Convex, ECOS
using JLD2
using StaticArrays
import DCD
import MeshCat as mc


P1 = DCD.Sphere(1.3)
P2 = DCD.Sphere(0.9)

P1.r = 2*(@SVector randn(3))
# P1.q = normalize((@SVector randn(4)))
P2.r = 1*(@SVector randn(3))
# P2.q = normalize((@SVector randn(4)))

# @inline function problem_matrices(sphere::DCD.Sphere{T},r::SVector{3,T1},any_attitude::SVector{n,T2}) where {T,T1,T2,n}
#     h_ort = SArray{Tuple{0}, Float64, 1, 0}(())
#     G_ort = SArray{Tuple{0, 4}, Float64, 2, 0}(())
#
#     h_soc = [0;-r]
#     # G_soc = [0 0 0 -P1.R; -I(3) zeros(3)]
#     # G_soc_top = [0 0 0 -sphere.R]
#     G_soc = SA[
#                  0  0  0 -sphere.R
#                 -1  0  0  0;
#                  0 -1  0  0;
#                  0  0 -1  0
#                  ]
#     # G_soc = [G_soc_top; G_soc_bot]
#
#     return G_ort, h_ort, G_soc, h_soc
# end

x = Variable(3)
α = Variable()


prob = minimize(α)
prob.constraints += α >= 0
prob.constraints += norm(x - P1.r) <= α*P1.R
prob.constraints += norm(x - P2.r) <= α*P2.R

solve!(prob, ECOS.Optimizer)
α = α.value
x = vec(x.value)

α2, x2 = DCD.proximity(P1,P2)
# α2, x2, J = DCD.proximity_jacobian(P1,P2)
# @btime DCD.proximity($P1,$P2)
# @btime DCD.proximity_jacobian($P1,$P2)
# # second problem
# x2 = Variable(3)
# α2 = Variable()
# s1 = Variable(4)
# s2 = Variable(4)
#
# prob2 = minimize(α2)
# prob2.constraints += α2 >= 0
# G_ort1,h_ort1,G1,h1 = problem_matrices(P1,P1.r,P1.q)
# s1 = h1 - G1*[x2;α2]
# prob2.constraints += norm(s1[2:end]) <= s1[1]
# G_ort2,h_ort2,G2,h2 = problem_matrices(P2,P2.r,P2.q)
# s2 = h2 - G2*[x2;α2]
# prob2.constraints += norm(s2[2:end]) <= s2[1]
#
# c,G,h,idx_ort,idx_soc1,idx_soc2 = DCD.combine_problem_matrices(G_ort1,h_ort1,G1,h1,G_ort2,h_ort2,G2,h2)
#
# x3,s3,z3 = DCD.solve_socp(c,G,h,idx_ort,idx_soc1,idx_soc2; verbose = false)
#
# @btime DCD.solve_socp($c,$G,$h,$idx_ort,$idx_soc1,$idx_soc2; verbose = false)
#
# # @btime DCD.combine_problem_matrices($G_ort1,$h_ort1,$G1,$h1,$G_ort2,$h_ort2,$G2,$h2)
# solve!(prob2, ECOS.Optimizer)
#
# α2 = α2.value
# x2 = vec(x2.value)

# α_star, x_star = DCD.proximity(P1,P2)
# @btime DCD.proximity($P1,$P2)

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
# @test norm(x - x_star,1) < 1e-4
# @test abs(α - α_star) < 1e-4

using Test
@test norm(x-x2) <= 1e-4
@test abs(α - α2) <= 1e-4
