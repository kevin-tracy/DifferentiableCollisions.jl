cd("/Users/kevintracy/.julia/dev/DCD/extras")
import Pkg; Pkg.activate(".")
import DCD
using LinearAlgebra, StaticArrays
import MeshCat as mc
using JLD2
using Convex
import ECOS
import Random
using Colors
Random.seed!(1)

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


trans_1 = 1.0
# trans_2 = 0.5


polytope = DCD.Polytope(SMatrix{8,3}(A2),SVector{8}(b2))
# polytope.q = normalize((@SVector randn(4)))
polytope.r = SA[0,0,2.0]

capsule = DCD.Capsule(0.5,1.2)
capsule.r = SA[0,0,2.0]

cylinder = DCD.Cylinder(0.5,1.2)
cylinder.r = SA[0,0,2.0]

cone = DCD.Cone(2.0,deg2rad(22))
cone.r = SA[0,0,2.0]
cone.q = SA[cos(-pi/4),0,sin(-pi/4),0]
# cone.q = normalize((@SVector randn(4)))

sphere = DCD.Sphere(0.8)
sphere.r = SA[0,0,2.0]

polygon = DCD.Polygon(create_n_sided(5,0.6)...,0.2)
polygon.r = SA[0.0,0,2.0]
polygon.q = SA[cos(pi/4),sin(pi/4),0,0]


# DCD.build_primitive!(vis, polytope, :polytope; α = 1.0,color = mc.RGBA(c1..., trans_1))
# DCD.build_primitive!(vis, capsule,  :capsule; α = 1.0,color = mc.RGBA(c2..., trans_1))
# DCD.build_primitive!(vis, cylinder, :cylinder; α = 1.0,color = mc.RGBA(c2..., trans_1))
# DCD.build_primitive!(vis, cone,     :cone; α = 1.0,color = mc.RGBA(c2..., trans_1))
# DCD.build_primitive!(vis, sphere,   :sphere; α = 1.0,color = mc.RGBA(c2..., trans_1))
# DCD.build_primitive!(vis, polygon,  :polygon; α = 1.0,color = mc.RGBA(c2..., trans_1))


# DCD.update_pose!(vis[:polytope], polytope)
# DCD.update_pose!(vis[:capsule],  capsule)
# DCD.update_pose!(vis[:cylinder], cylinder)
# DCD.update_pose!(vis[:cone],     cone)
# DCD.update_pose!(vis[:sphere],   sphere)
# DCD.update_pose!(vis[:polygon],  polygon)




DCD.set_floor!(vis)

N_αs = 30
αs = [range(.05,1.0,N_αs);reverse(range(.05,1.0,N_αs))]

object_names = []

for i = 1:length(αs)
    name = Symbol("polytope"*string(i))
    push!(object_names, name)
    DCD.build_primitive!(vis, polytope, name; α = αs[i],color = mc.RGBA(c1..., trans_1))
    DCD.update_pose!(vis[name],polytope)
    mc.setvisible!(vis[name], false)
end
for i = 1:length(αs)
    name = Symbol("capsule"*string(i))
    push!(object_names, name)
    DCD.build_primitive!(vis, capsule, name; α = αs[i],color = mc.RGBA(c1..., trans_1))
    DCD.update_pose!(vis[name], capsule)
    mc.setvisible!(vis[name], false)
end
for i = 1:length(αs)
    name = Symbol("cylinder"*string(i))
    push!(object_names, name)
    DCD.build_primitive!(vis, cylinder, name; α = αs[i],color = mc.RGBA(c1..., trans_1))
    DCD.update_pose!(vis[name], cylinder)
    mc.setvisible!(vis[name], false)
end
for i = 1:length(αs)
    name = Symbol("cone"*string(i))
    push!(object_names, name)
    DCD.build_primitive!(vis, cone, name; α = αs[i],color = mc.RGBA(c1..., trans_1))
    DCD.update_pose!(vis[name], cone)
    mc.setvisible!(vis[name], false)
end
for i = 1:length(αs)
    name = Symbol("sphere"*string(i))
    push!(object_names, name)
    DCD.build_primitive!(vis, sphere, name; α = αs[i],color = mc.RGBA(c1..., trans_1))
    DCD.update_pose!(vis[name], sphere)
    mc.setvisible!(vis[name], false)
end
for i = 1:length(αs)
    name = Symbol("polygon"*string(i))
    push!(object_names, name)
    DCD.build_primitive!(vis, polygon, name; α = αs[i],color = mc.RGBA(c1..., trans_1))
    DCD.update_pose!(vis[name], polygon)
    mc.setvisible!(vis[name], false)
end

mc.setprop!(vis["/Background"], "top_color", colorant"transparent")
DCD.set_floor!(vis)
anim = mc.Animation(floor(Int,1/0.04))

for k = 1:length(object_names)
    mc.atframe(anim, k) do
        for i = 1:length(object_names)
            mc.setvisible!(vis[object_names[i]], false)
        end
        mc.setvisible!(vis[object_names[k]], true)
    end
end
mc.setanimation!(vis, anim)

# αs = 1:.05:2.5
# rαs = reverse(αs)
# cone_names = [Symbol("cone"*string(i)) for i = 1:length(αs)]
# polytope_names = [Symbol("polytope"*string(i)) for i = 1:length(αs)]
# for i = 1:length(αs)
#     DCD.build_primitive!(vis, cone, cone_names[i]; α = αs[i],color = mc.RGBA(c2..., trans_2))
#     DCD.update_pose!(vis[cone_names[i]],cone)
#     DCD.build_primitive!(vis, polytope, polytope_names[i]; α = αs[i],color = mc.RGBA(c1..., trans_2))
#     DCD.update_pose!(vis[polytope_names[i]],polytope)
#     mc.setvisible!(vis[cone_names[i]], false)
#     mc.setvisible!(vis[polytope_names[i]], false)
# end
#
#
# ambient=0.55
# direction="Positive"
#
# mc.setprop!(vis["/Lights/AmbientLight/<object>"], "intensity", ambient)
# mc.setprop!(vis["/Lights/FillLight/<object>"], "intensity", 0.25)
# mc.setprop!(vis["/Lights/PointLight$(direction)Y/<object>"], "intensity", 0.85)
# mc.setprop!(vis["/Lights/PointLight$(direction)Y/<object>"], "castShadow", true)
#
# # mc.setprop!(vis["/Background"], "top_color", color = mc.RGBA(1,0,0.0, 1.0))
# mc.setprop!(vis["/Background"], "top_color", colorant"transparent")
#
# anim = mc.Animation(floor(Int,1/0.1))
#
# for k = 1:(2*length(αs) - 1)
#     mc.atframe(anim, k) do
#         if k <= length(αs)
#             for i = 1:length(αs)
#                 mc.setvisible!(vis[cone_names[i]], false)
#                 mc.setvisible!(vis[polytope_names[i]], false)
#             end
#             mc.setvisible!(vis[cone_names[k]], true)
#             mc.setvisible!(vis[polytope_names[k]], true)
#         else
#             for i = 1:length(αs)
#                 mc.setvisible!(vis[cone_names[i]], false)
#                 mc.setvisible!(vis[polytope_names[i]], false)
#             end
#             idx = k - 2*(k -  length(αs))
#             mc.setvisible!(vis[cone_names[idx]], true)
#             mc.setvisible!(vis[polytope_names[idx]], true)
#         end
#     end
# end
# mc.setanimation!(vis, anim)
