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
@load "/Users/kevintracy/.julia/dev/DifferentialProximity/extras/polyhedra_plotting/polytopes.jld2"

b2 = 0.7*b2

function solve_intersect(cone,polytope)
    x = Variable(3)
    α = Variable()

    p = minimize(α)
    p.constraints += α >= 0
    p.constraints += polytope.A*(DCD.dcm_from_q(polytope.q)'*(x - polytope.r)) <= α*polytope.b
    # p.constraints += A2*(Q2'*(x - r2)) <= α*b2

    Q2 = DCD.dcm_from_q(cone.q)
    bx = Q2*[1,0,0]
    c = cone.r - α*(cone.H*3/4)*bx
    d = cone.r + α*(cone.H/4)*bx
    x̃ = Q2'*(x - c)
    p.constraints += norm(x̃[2:3]) <= tan(cone.β)*x̃[1]
    p.constraints += (x - d)'*bx <= 0
    solve!(p, ECOS.Optimizer)
    # α = α.value
    return vec(x.value), α.value
end

vis = mc.Visualizer()
open(vis)

c1 = [245, 155, 66]/255
c2 = [2,190,207]/255


trans_1 = 1.0
trans_2 = 0.5
cone = DCD.Cone(2.0,deg2rad(22))
cone.r = SA[-1,0,2.0]
cone.q = normalize((@SVector randn(4)))

polytope = DCD.Polytope(SMatrix{8,3}(A2),SVector{8}(b2))
polytope.q = normalize((@SVector randn(4)))
polytope.r = SA[2,2,2.0]

DCD.build_primitive!(vis, cone, :cone; α = 1.0,color = mc.RGBA(c2..., trans_1))
DCD.update_pose!(vis[:cone],cone)
DCD.build_primitive!(vis, polytope, :polytope; α = 1.0,color = mc.RGBA(c1..., trans_1))
DCD.update_pose!(vis[:polytope],polytope)

DCD.set_floor!(vis; darkmode = false)
DCD.add_axes!(vis,:Nframe, 1.0, 0.02; head_l = 0.1, head_w = 0.05)

# x_star, α_star = solve_intersect(cone,polytope)

α_star,x_star = DCD.proximity(cone,polytope)

DCD.build_primitive!(vis, cone, :cone2; α = α_star,color = mc.RGBA(c2..., trans_2))
DCD.update_pose!(vis[:cone2],cone)
DCD.build_primitive!(vis, polytope, :polytope2; α = α_star,color = mc.RGBA(c1..., trans_2))
DCD.update_pose!(vis[:polytope2],polytope)
ambient=0.55
direction="Positive"

mc.setprop!(vis["/Lights/AmbientLight/<object>"], "intensity", ambient)
mc.setprop!(vis["/Lights/FillLight/<object>"], "intensity", 0.25)
mc.setprop!(vis["/Lights/PointLight$(direction)X/<object>"], "intensity", 0.85)
mc.setprop!(vis["/Lights/PointLight$(direction)X/<object>"], "castShadow", true)
# αs = range(1,α_star,30)
# cone_names = [Symbol("cone"*string(i)) for i = 1:length(αs)]
# polytope_names = [Symbol("polytope"*string(i)) for i = 1:length(αs)]
# for i = length(αs)
#     DCD.build_primitive!(vis, cone, cone_names[i]; α = αs[i],color = mc.RGBA(c2..., trans_2))
#     DCD.update_pose!(vis[cone_names[i]],cone)
#     DCD.build_primitive!(vis, polytope, polytope_names[i]; α = αs[i],color = mc.RGBA(c1..., trans_2))
#     DCD.update_pose!(vis[polytope_names[i]],polytope)
#     # mc.setvisible!(vis[cone_names[i]], false)
#     # mc.setvisible!(vis[polytope_names[i]], false)
# end


# ambient=0.55
# direction="Positive"
#
# mc.setprop!(vis["/Lights/AmbientLight/<object>"], "intensity", ambient)
# mc.setprop!(vis["/Lights/FillLight/<object>"], "intensity", 0.25)
# mc.setprop!(vis["/Lights/PointLight$(direction)X/<object>"], "intensity", 0.85)
# mc.setprop!(vis["/Lights/PointLight$(direction)X/<object>"], "castShadow", true)

# mc.setprop!(vis["/Background"], "top_color", color = mc.RGBA(1,0,0.0, 1.0))
# mc.setprop!(vis["/Background"], "top_color", colorant"transparent")
#
# anim = mc.Animation(floor(Int,1/0.1))
#
sph_p1 = mc.HyperSphere(mc.Point(x_star...), 0.1)
mc.setobject!(vis[:pooint1], sph_p1,mc.MeshPhongMaterial(color = mc.RGBA(1.0,0,0,1.0)))
#
#
# # for k = 1:(2*length(αs))
#     mc.atframe(anim, k) do
#         if k <= length(αs)
#             for i = 1:length(αs)
#                 mc.setvisible!(vis[cone_names[i]], false)
#                 mc.setvisible!(vis[polytope_names[i]], false)
#             end
#             mc.setvisible!(vis[:pooint1], false)
#             mc.setvisible!(vis[cone_names[k]], true)
#             mc.setvisible!(vis[polytope_names[k]], true)
#         else
#             mc.setvisible!(vis[:pooint1], true)
#         end
#     end
# end
# mc.setanimation!(vis, anim)
