using Pkg
Pkg.activate(dirname(@__DIR__))
# import DifferentialProximity as dp
import DCOL as dc
using LinearAlgebra
# using MeshCat, GeometryBasics, CoordinateTransformations, Rotations
# using Colors
# using Polyhedra
import MeshCat as mc
using StaticArrays
using JLD2
# using Convex, ECOS
# include("visuals.jl")
# include("poly_plotting.jl")

@load "/Users/kevintracy/.julia/dev/DifferentialProximity/extras/polyhedra_plotting/polytopes.jld2"

vis = mc.Visualizer()
open(vis)

r1= [2;2;3.2]
r2 = [0;0;2.2]
import Random
Random.seed!(25)
q1 = SVector{4}(normalize(randn(4)))
q2 = SVector{4}(normalize(randn(4)))
q1 = SVector{4}(normalize(randn(4)))
c1 = [245, 155, 66]/255
c2 = [2,190,207]/255
trans_1 = 1.0
trans_2 = 0.3
polytope = dc.Polytope(SMatrix{14,3}(A1),SVector{14}(b1))

H = 2.3
β = deg2rad(21)
# W = H*tan(β)
cone = dc.Cone(H,β)
cone.r = r2
cone.q = q2

polytope.r = r1
polytope.q = q1

# build_polytope!(A1,b1,vis,:polytope1_small;color = Colors.RGBA((c1/255)..., first_trans))
# build_polytope!(A1,α*b1,vis,:polytope1_big;color = Colors.RGBA((c1/255)..., second_trans))
dc.build_primitive!(vis, cone, :cone; α = 1.0,color = mc.RGBA(c2..., trans_1))
dc.update_pose!(vis[:cone],cone)
dc.build_primitive!(vis, polytope, :polytope; α = 1.0,color = mc.RGBA(c1..., trans_1))
dc.update_pose!(vis[:polytope],polytope)

dc.set_floor!(vis; darkmode = false, x = 40, y = 40)
dc.add_axes!(vis,:Nframe, 1.0, 0.02; head_l = 0.1, head_w = 0.05)

# x_star, α_star = solve_intersect(cone,polytope)

α_star,x_star = dc.proximity(cone,polytope)

dc.build_primitive!(vis, cone, :cone2; α = α_star,color = mc.RGBA(c2..., trans_2))
dc.update_pose!(vis[:cone2],cone)
dc.build_primitive!(vis, polytope, :polytope2; α = α_star,color = mc.RGBA(c1..., trans_2))
dc.update_pose!(vis[:polytope2],polytope)

# dc.add_axes!(vis,:cone_axes,)

# # build_cone(vis, :cone_small, H, W, Colors.RGBA((c2/255)..., first_trans))
# # build_cone(vis, :cone_big, α*H, α*W, Colors.RGBA((c2/255)..., second_trans))
# # build_polytope!(A2,b2,vis,:polytope2_small;color = Colors.RGBA((c2/255)..., first_trans))
# # build_polytope!(A2,α*b2,vis,:polytope2_big;color = Colors.RGBA((c2/255)..., second_trans))
#
#
# settransform!(vis[:polytope1_small],Translation(r1) ∘ LinearMap(QuatRotation(q1)))
# settransform!(vis[:polytope1_big],Translation(r1) ∘ LinearMap(QuatRotation(q1)))
# # r2 = [1,0,0.]
# # q2 = [1,0,0,0.0]
# settransform!(vis[:cone_small],Translation(r2) ∘ LinearMap(QuatRotation(q2)))
# settransform!(vis[:cone_big],Translation(r2) ∘ LinearMap(QuatRotation(q2)))

ambient=0.84
direction="Positive"

mc.setprop!(vis["/Lights/AmbientLight/<object>"], "intensity", ambient)
mc.setprop!(vis["/Lights/FillLight/<object>"], "intensity", 0.0)
mc.setprop!(vis["/Lights/PointLight$(direction)X/<object>"], "intensity", 0.85)
mc.setprop!(vis["/Lights/PointLight$(direction)X/<object>"], "castShadow", false)
mc.setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 3.0)
mc.setvisible!(vis["/Background"], false )
mc.setvisible!(vis["/Grid"], false )
# set_floor!(vis)
#
sph_a = mc.HyperSphere(mc.Point(x_star...), 0.1)
# intersect_contact_point = vec(x.value)
mc.setobject!(vis[:cp],sph_a, mc.MeshPhongMaterial(color=mc.RGBA(1,0,0.0, 1.0)))
# settransform!(vis[:cp], Translation(intersect_contact_point))
#
# dc.add_axes!(vis,:axez1, 1.2, 0.02; head_l = 0.1, head_w = 0.05)
# mc.settransform!(vis[:axez1],mc.Translation(r1) ∘ mc.LinearMap(dc.dcm_from_q(q1)))
# #
# dc.add_axes!(vis,:axez2, 2.0, 0.02; head_l = 0.1, head_w = 0.05)
# mc.settransform!(vis[:axez2],mc.Translation(r2) ∘ mc.LinearMap(dc.dcm_from_q(q2)))
#
# # add_axes!(vis,:axez_world, 0.5, 0.02; head_l = 0.1, head_w = 0.05)
#
# dc.add_axes!(vis,:axes_W, 0.6, 0.02; head_l = 0.1, head_w = 0.05)
