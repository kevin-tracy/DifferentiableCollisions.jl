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

DCD.proximity(P1,P2)
using BenchmarkTools
@btime DCD.proximity($P1,$P2; pdip_tol = 1e-6)

DCD.proximity_jacobian(P1,P2)
@btime DCD.proximity_jacobian($P1,$P2; pdip_tol = 1e-6)
# test if solver can solve LP's
# G_ort1, h_ort1, G_soc1, h_soc1 = DCD.problem_matrices(P1,P1.r,P1.q)
# G_ort2, h_ort2, G_soc2, h_soc2 = DCD.problem_matrices(P2,P2.r,P2.q)
# c,G,h,idx_ort,idx_soc1,idx_soc2 = DCD.combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1,G_ort2, h_ort2, G_soc2, h_soc2)
# x,s,z = DCD.solve_socp(c,G,h,idx_ort,idx_soc1,idx_soc2; verbose = true, pdip_tol = 1e-8)
#
# @btime DCD.solve_socp($c,$G,$h,$idx_ort,$idx_soc1,$idx_soc2; verbose = false, pdip_tol = 1e-6)
#
# x = Variable(3)
# α = Variable()
#
# Q1 = DCD.dcm_from_q(P1.q)
# Q2 = DCD.dcm_from_q(P2.q)
#
# prob = minimize(α)
# prob.constraints += α >= 0
# prob.constraints += P1.A*Q1'*(x - P1.r) <= P1.b*α
# prob.constraints += P2.A*Q2'*(x - P2.r) <= P2.b*α
#
# solve!(prob, ECOS.Optimizer)
#
# α = α.value
# x = vec(x.value)
#
# α_star, x_star = DCD.proximity(P1,P2)
# # @btime DCD.proximity($P1,$P2)
#
# vis = mc.Visualizer()
# open(vis)
#
# c1 = [245, 155, 66]/255
# c2 = [2,190,207]/255
#
#
# if α > 1
#     trans_1 = 1.0
#     trans_2 = 0.5
# else
#     trans_2 = 1.0
#     trans_1 = 0.5
# end
#
# DCD.build_primitive!(vis, P1, :polytope1; α = 1.0,color = mc.RGBA(c1..., trans_1))
# DCD.build_primitive!(vis, P2, :polytope2; α = 1.0,color = mc.RGBA(c2..., trans_1))
# DCD.build_primitive!(vis, P1, :polytope1_big; α = α,color = mc.RGBA(c1..., trans_2))
# DCD.build_primitive!(vis, P2, :polytope2_big; α = α,color = mc.RGBA(c2..., trans_2))
#
# DCD.update_pose!(vis[:polytope1], P1)
# DCD.update_pose!(vis[:polytope2], P2)
# DCD.update_pose!(vis[:polytope1_big], P1)
# DCD.update_pose!(vis[:polytope2_big], P2)
#
# sph_p1 = mc.HyperSphere(mc.Point(x...), 0.1)
# mc.setobject!(vis[:p1], sph_p1, mc.MeshPhongMaterial(color = mc.RGBA(1.0,0,0,1.0)))
#
# @info α
# @test norm(x - x_star,1) < 1e-4
# @test abs(α - α_star) < 1e-4
