cd("/Users/kevintracy/.julia/dev/DCOL/extras")
import Pkg; Pkg.activate(".")
import DCOL as DCD
using LinearAlgebra, StaticArrays
import MeshCat as mc
using JLD2
using Convex
import ECOS
import Random
using Colors
Random.seed!(1)
function qdot(q1,q2)
        s1 = q1[1]
        v1 = q1[2:4]
        s2 = q2[1]
        v2 = q2[2:4]
        [s1*s2 - dot(v1,v2); s2*v1 + s1*v2 + cross(v1,v2)]
end

function q_from_aa(r,θ)
    [cos(θ/2); r*sin(θ/2)]
end
function create_n_sided(N,d)
    ns = [ [cos(θ);sin(θ)] for θ = 0:(2*π/N):(2*π*(N-1)/N)]
    A = vcat(transpose.((ns))...)
    b = d*ones(N)
    return SMatrix{N,2}(A), SVector{N}(b)
end

@load "/Users/kevintracy/.julia/dev/DifferentialProximity/extras/polyhedra_plotting/polytopes.jld2"
b2 = 0.7*b2


vis = mc.Visualizer()
open(vis)

c1 = [245, 155, 66]/255
c2 = [2,190,207]/255


trans_1 = 0.9
# trans_2 = 0.5

import Random
Random.seed!(20)
zoff = SA[0,0,-1]

polytope = DCD.Polytope(SMatrix{8,3}(A2),SVector{8}(b2))
polytope.q = normalize((@SVector randn(4)))
polytope.r = [1,1,3.0] + zoff

capsule = DCD.Capsule(0.6,2.0)
capsule.r = [1, 1, 3]+ zoff
capsule.q = normalize((@SVector randn(4)))

cylinder = DCD.Cylinder(0.4,1.5)
cylinder.r = [1,1,2.5]+ zoff
cylinder.q = copy(capsule.q)

cone = DCD.Cone(2.0,deg2rad(22))
cone.r = [1.1,1.1,2.5]+ zoff
cone.q = qdot(capsule.q, q_from_aa([1,0,0],pi/2))
# cone.q = normalize((@SVector randn(4)))

sphere = DCD.Sphere(0.7)
sphere.r = [1,1,2.5]+ zoff

polygon = DCD.Polygon(create_n_sided(5,0.95)...,0.1)
polygon.r = SA[1,1,3.0]+ zoff
polygon.q = qdot(capsule.q, q_from_aa([0,1,0],deg2rad(80)))


# DCD.build_primitive!(vis, polytope, :polytope; α = 1.0,color = mc.RGBA(c1..., trans_1))
# DCD.build_primitive!(vis, capsule,  :capsule; α = 1.0,color = mc.RGBA(c1..., trans_1))
# DCD.build_primitive!(vis, cylinder, :cylinder; α = 1.0,color = mc.RGBA(c1..., trans_1))
# DCD.build_primitive!(vis, cone,     :cone; α = 1.0,color = mc.RGBA(c1..., trans_1))
# DCD.build_primitive!(vis, sphere,   :sphere; α = 1.0,color = mc.RGBA(c1..., trans_1))
DCD.build_primitive!(vis, polygon,  :polygon; α = 1.0,color = mc.RGBA(c1..., trans_1))


# DCD.update_pose!(vis[:polytope], polytope)
# DCD.update_pose!(vis[:capsule],  capsule)
# DCD.update_pose!(vis[:cylinder], cylinder)
# DCD.update_pose!(vis[:cone],     cone)
# DCD.update_pose!(vis[:sphere],   sphere)
DCD.update_pose!(vis[:polygon],  polygon)

# DCD.add_axes!(vis,:axes_polytope, 1.2, 0.02; head_l = 0.1, head_w = 0.05)
# DCD.add_axes!(vis,:axes_capsule, 2.0, 0.02; head_l = 0.1, head_w = 0.05)
# DCD.add_axes!(vis,:axes_cylinder, 1.2, 0.02; head_l = 0.1, head_w = 0.05)
# DCD.add_axes!(vis,:axes_cone, 1.1, 0.02; head_l = 0.1, head_w = 0.05)
# DCD.add_axes!(vis,:axes_sphere, 1.2, 0.02; head_l = 0.1, head_w = 0.05)
DCD.add_axes!(vis,:axes_polygon, 1.4, 0.02; head_l = 0.1, head_w = 0.05)

# DCD.update_pose!(vis[:axes_polytope], polytope)
# DCD.update_pose!(vis[:axes_capsule],  capsule)
# DCD.update_pose!(vis[:axes_cylinder], cylinder)
# DCD.update_pose!(vis[:axes_cone],     cone)
# DCD.update_pose!(vis[:axes_sphere],   sphere)
DCD.update_pose!(vis[:axes_polygon],  polygon)

DCD.add_axes!(vis,:axes_W, 0.6, 0.02; head_l = 0.1, head_w = 0.05)


mc.setprop!(vis["/Lights/AmbientLight/<object>"], "intensity", 0.9)
mc.setprop!(vis["/Lights/PointLightPositiveX/<object>"], "intensity", 0.0)
mc.setprop!(vis["/Lights/FillLight/<object>"], "intensity", 0.25)
# mc.setprop!(vis["/Background"], "top_color", colorant"transparent")
mc.setvisible!(vis["/Grid"],false)
mc.setvisible!(vis["/Background"],false)
mc.setvisible!(vis["/Axes"],false)

DCD.set_floor!(vis; x = 20, y = 20, darkmode = false)
