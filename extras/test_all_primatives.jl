import Pkg; Pkg.activate("/Users/kevintracy/.julia/dev/DCD/extras")

using LinearAlgebra
using Convex, ECOS
using JLD2
using StaticArrays
import DCD
import MeshCat as mc

function create_n_sided(N,d)
    ns = [ [cos(θ);sin(θ)] for θ = 0:(2*π/N):(2*π*(N-1)/N)]
    A = vcat(transpose.((ns))...)
    b = d*ones(N)
    return SMatrix{N,2}(A), SVector{N}(b)
end

@load "/Users/kevintracy/.julia/dev/DifferentialProximity/extras/polyhedra_plotting/polytopes.jld2"
b2 = 0.7*b2

P1 = DCD.Polytope(SMatrix{8,3}(A2),SVector{8}(b2))
# P1 = DCD.Capsule(0.5,1.2)
# P1 = DCD.Cylinder(0.5,1.2)
# P1 = DCD.Cone(2.0,deg2rad(22))
# P1 = DCD.Sphere(0.8)
# P1 = DCD.Polygon(create_n_sided(5,0.6)...,0.2)

# P2 = DCD.Polytope(SMatrix{14,3}(A1),SVector{14}(b1))
# P2 = DCD.Capsule(0.7,1.4)
# P2 = DCD.Cylinder(0.4,1.2)
# P2 = DCD.Cone(2.2,deg2rad(18))
# P2 = DCD.Sphere(0.63)
P2 = DCD.Polygon(create_n_sided(8,0.7)...,0.16)

P1.r = 1*(@SVector randn(3))
P1.q = normalize((@SVector randn(4)))
P2.r = 1*(@SVector randn(3))
P2.q = normalize((@SVector randn(4)))


α,x,J = DCD.proximity_jacobian(P1,P2)

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
