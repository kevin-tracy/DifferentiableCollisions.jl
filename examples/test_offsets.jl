using Pkg
Pkg.activate(joinpath((@__DIR__), ".."))
using DCOL
Pkg.activate((@__DIR__))
# Pkg.instantiate()

using LinearAlgebra
using StaticArrays
import MeshCat as mc
import DCOL as dc
import Random
using JLD2
# using Colors
const DCD = dc

# Random.seed!(1)

function qdot(q1,q2)
        s1 = q1[1]
        v1 = q1[2:4]
        s2 = q2[1]
        v2 = q2[2:4]
        [s1*s2 - dot(v1,v2); s2*v1 + s1*v2 + cross(v1,v2)]
end

function skew(v)
    SA[0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
end
function dcm_from_phi(r,θ)
    K = skew(r)
    I + sin(θ)*K + (1 - cos(θ))*K^2
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


# vis = mc.Visualizer()
# open(vis)

c1 = [245, 155, 66]/255
c2 = [2,190,207]/255


P_obs = dc.Polygon(create_n_sided(10,1.1)...,0.3)
P_obs.r = (@SVector randn(3))
P_obs.q = normalize((@SVector randn(4)))

trans_1 = 0.9
# trans_2 = 0.5

# import Random
# Random.seed!(20)
zoff = SA[0,0,-1]

P = []
push!(P,dc.Polytope(SMatrix{8,3}(A2),SVector{8}(b2)))
# P[1].q = normalize((@SVector randn(4)))
# P[1].r = [1,1,3.0] + zoff
# P[1].Q_offset = dc.dcm_from_q(normalize((@SVector randn(4))))
P[1].r_offset = SA[0,0,3.0]

push!(P,dc.Capsule(0.5,2.0))
P[2].Q_offset = dcm_from_phi(SA[0,1,0.0], -deg2rad(90))
# capsule = dc.Capsule(0.5,2.0)
# capsule.r = [1, 1, 3]+ zoff
# capsule.q = normalize((@SVector randn(4)))

push!(P,dc.Cylinder(0.4,2.3))
P[3].r_offset = SA[0,-2,0.0]
P[3].Q_offset = dcm_from_phi(SA[0,0,1.0], deg2rad(90))*dcm_from_phi(SA[0,0,1.0], deg2rad(20))

push!(P,dc.Cylinder(0.4,2.3))
P[4].r_offset = SA[0,2,0.0]
P[4].Q_offset = dcm_from_phi(SA[0,0,1.0], -deg2rad(90))*dcm_from_phi(SA[0,0,1.0], -deg2rad(20))

push!(P, dc.Cone(2.0,deg2rad(22)))
P[5].r_offset = SA[0,0,-2.5]
P[5].Q_offset = dcm_from_phi(SA[0,1,0.0], -deg2rad(90))

push!(P, dc.Ellipsoid(SMatrix{3,3}(Diagonal((2.5*[1,1/2,1/3]) .^ 2))))
P[6].r_offset = SA[1.0,0,-2.5]
P[6].Q_offset = dcm_from_phi(SA[0,1,0.0], -deg2rad(45))

da_r = @SVector randn(3)
da_q = normalize(@SVector randn(4))
for i = 1:length(P)
    P[i].r = 1*da_r
    P[i].q = 1*da_q
end
#
# cylinder = dc.Cylinder(0.5,2.0)
# cylinder.r = [1,1,2.5]+ zoff
# cylinder.q = copy(capsule.q)
#
# cone = dc.Cone(2.0,deg2rad(22))
# cone.r = [1.1,1.1,2.5]+ zoff
# cone.q = qdot(capsule.q, q_from_aa([1,0,0],pi/2))
# # cone.q = normalize((@SVector randn(4)))
#
# sphere = dc.Sphere(0.7)
# sphere.r = [1,1,2.5]+ zoff
#
# ellipse = dc.Ellipsoid(SMatrix{3,3}(Diagonal((2.5*[1,1/2,1/3]) .^ 2)))
# ellipse.r = 1*sphere.r
# ellipse.q = 1*capsule.q
#
# polygon = dc.Polygon(create_n_sided(5,0.95)...,0.1)
# polygon.r = SA[1,1,3.0]+ zoff
# polygon.q = qdot(capsule.q, q_from_aa([0,1,0],deg2rad(80)))

for i = 1:length(P)
    if dc.proximity(P_obs,P[i])[1] >= 1
        dc.build_primitive!(vis, P[i], "P"*string(i); α = 1.0,color = mc.RGBA(c1..., trans_1))
    else
        dc.build_primitive!(vis, P[i], "P"*string(i); α = 1.0,color = mc.RGBA(1,0,0.0, trans_1))
    end
    dc.update_pose!(vis["P"*string(i)],P[i])
end
dc.build_primitive!(vis, P_obs, :obs; α = 1.0, color = mc.RGBA(c2...,trans_1))
dc.update_pose!(vis[:obs], P_obs)
# dc.build_primitive!(vis, polytope, :polytope; α = 1.0,color = mc.RGBA(c1..., trans_1))
# # # DCD.build_primitive!(vis, capsule,  :capsule; α = 1.0,color = mc.RGBA(c1..., trans_1))
# # # DCD.build_primitive!(vis, cylinder, :cylinder; α = 1.0,color = mc.RGBA(c1..., trans_1))
# # # DCD.build_primitive!(vis, cone,     :cone; α = 1.0,color = mc.RGBA(c1..., trans_1))
# # # DCD.build_primitive!(vis, sphere,   :sphere; α = 1.0,color = mc.RGBA(c1..., trans_1))
# # dc.build_primitive!(vis, ellipse,   :ellipse; α = 1.0,color = mc.RGBA(c1..., trans_1))
# # DCD.build_primitive!(vis, polygon,  :polygon; α = 1.0,color = mc.RGBA(c1..., trans_1))
# #
# #
# DCD.update_pose!(vis[:polytope], polytope)
# # # DCD.update_pose!(vis[:capsule],  capsule)
# # # DCD.update_pose!(vis[:cylinder], cylinder)
# # # DCD.update_pose!(vis[:cone],     cone)
# # # DCD.update_pose!(vis[:sphere],   sphere)
# # # DCD.update_pose!(vis[:polygon],  polygon)
# # dc.update_pose!(vis[:ellipse], ellipse)
# #
for i = 1:length(P)
    dc.add_axes!(vis, "axes"*string(i),1.2,0.02; head_l = 0.1, head_w = 0.05)
    dc.update_pose!(vis["axes"*string(i)], P[i])
end
dc.add_axes!(vis, :obs_axes,1.2,0.02; head_l = 0.1, head_w = 0.05)
dc.update_pose!(vis[:obs_axes], P_obs)
# DCD.add_axes!(vis,:axes_polytope, 1.2, 0.02; head_l = 0.1, head_w = 0.05)
# # # DCD.add_axes!(vis,:axes_capsule, 2.0, 0.02; head_l = 0.1, head_w = 0.05)
# # # DCD.add_axes!(vis,:axes_cylinder, 1.2, 0.02; head_l = 0.1, head_w = 0.05)
# # # DCD.add_axes!(vis,:axes_cone, 1.1, 0.02; head_l = 0.1, head_w = 0.05)
# # # DCD.add_axes!(vis,:axes_sphere, 1.2, 0.02; head_l = 0.1, head_w = 0.05)
# # DCD.add_axes!(vis,:axes_polygon, 1.4, 0.02; head_l = 0.1, head_w = 0.05)
# # dc.add_axes!(vis,:axes_ellipse, 1.4, 0.02; head_l = 0.1, head_w = 0.05)
# #
# DCD.update_pose!(vis[:axes_polytope], polytope)
# # # DCD.update_pose!(vis[:axes_capsule],  capsule)
# # # DCD.update_pose!(vis[:axes_cylinder], cylinder)
# # # DCD.update_pose!(vis[:axes_cone],     cone)
# # # DCD.update_pose!(vis[:axes_sphere],   sphere)
# # # DCD.update_pose!(vis[:axes_polygon],  polygon)
# # dc.update_pose!(vis[:axes_ellipse],  ellipse)
# #
# dc.add_axes!(vis,:axes_W, 0.6, 0.02; head_l = 0.1, head_w = 0.05)
# #
# #
# mc.setprop!(vis["/Lights/AmbientLight/<object>"], "intensity", 0.9)
# mc.setprop!(vis["/Lights/PointLightPositiveX/<object>"], "intensity", 0.0)
# mc.setprop!(vis["/Lights/FillLight/<object>"], "intensity", 0.25)
# # mc.setprop!(vis["/Background"], "top_color", colorant"transparent")
# mc.setvisible!(vis["/Grid"],true)
# mc.setvisible!(vis["/Background"],false)
# mc.setvisible!(vis["/Axes"],false)
# mc.setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 3)
