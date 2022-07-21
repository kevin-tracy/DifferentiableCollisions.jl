# cd("/Users/kevintracy/.julia/dev/DCD/extras")
# import Pkg; Pkg.activate(".")

include("primitives.jl")
using LinearAlgebra, StaticArrays, Convex, ECOS
function dcm_from_q(q::SVector{4,T}) where {T}
    #DCM from quaternion, hamilton product, scalar first
    # pull our the parameters from the quaternion
    q4,q1,q2,q3 = normalize(q)

    # DCM
    Q = @SArray [(2*q1^2+2*q4^2-1)   2*(q1*q2 - q3*q4)   2*(q1*q3 + q2*q4);
          2*(q1*q2 + q3*q4)  (2*q2^2+2*q4^2-1)   2*(q2*q3 - q1*q4);
          2*(q1*q3 - q2*q4)   2*(q2*q3 + q1*q4)  (2*q3^2+2*q4^2-1)]
end

# collision between cone and capsule
vis1 = mc.Visualizer()
open(vis1)
vis2 = mc.Visualizer()
open(vis2)

cone = Cone(2.0,deg2rad(22))
cone.r = 0.3*(@SVector randn(3))
cone.q = normalize((@SVector randn(4)))
build_primitive!(vis1, cone, :cone; α = 1.0,color = mc.RGBA(0.1, 0.7, 0.7, 0.7))
update_pose!(vis1[:cone],cone)

capsule = Capsule(.3,1.2)
capsule.r = (@SVector randn(3))
capsule.q = normalize((@SVector randn(4)))
build_primitive!(vis1, capsule, :capsule; α = 1.0,color = mc.RGBA(0.7, 1.0, 0.7, 0.7))
update_pose!(vis1[:capsule],capsule)


# solve for collision
x = Variable(3)
α = Variable()
γ = Variable()

prob = minimize(α)
prob.constraints += α >= 0

# capsule constraints
n_Q_b_capsule = dcm_from_q(capsule.q)
bx = n_Q_b_capsule*[1,0,0]

# prob.constraints += norm(x - (capsule.r + γ*bx)) <= α*capsule.R
G_soc_caps= [0 0 0 -capsule.R 0;-diagm(ones(3)) zeros(3) bx]
h_soc_caps = [0; -(capsule.r)]
s = h_soc_caps - G_soc_caps*[x;α;γ]
prob.constraints += norm(s[2:4]) <= s[1]

# prob.constraints += γ <= α*capsule.L/2
# prob.constraints += -α*(capsule.L/2) <= γ
G_ort_caps = [0 0 0 -capsule.L/2 1; 0 0 0 -capsule.L/2 -1]
h_ort_caps = [0,0]
prob.constraints += G_ort_caps*[x;α;γ] <= h_ort_caps

# cone constraints
n_Q_b_cone = dcm_from_q(cone.q)
bx = n_Q_b_cone*[1,0,0]
c = cone.r - α*(cone.H/2)*bx
d = cone.r + α*(cone.H/2)*bx
x̃ = n_Q_b_cone'*(x - c)
prob.constraints += norm(x̃[2:3]) <= tan(cone.β)*x̃[1]
E = diagm([tan(cone.β),1,1])
h_soc_cone = -E*n_Q_b_cone'*cone.r
G_soc_cone = [(-E*n_Q_b_cone') (-[tan(cone.β)*cone.H/2,0,0]) zeros(3)]
# s2 = h_soc_cone - G_soc_cone*[x;α;γ]
# prob.constraints += norm(s2[2:3]) <= s2[1]

prob.constraints += (x - d)'*bx <= 0
G_ort_cone = [bx' -cone.H/2 0]
h_ort_cone = [dot(bx,cone.r)]
# prob.constraints += G_ort_cone*[x;α;γ] <= h_ort_cone

solve!(prob, ECOS.Optimizer)

x = vec(x.value)
α = α.value
γ = γ.value

z = Variable(5)
prob2 = minimize(z[4])
prob2.constraints += [G_ort_caps;G_ort_cone]*z <= [h_ort_caps;h_ort_cone]
s1 = h_soc_caps - G_soc_caps*z
s2 = h_soc_cone - G_soc_cone*z
prob2.constraints += norm(s1[2:end]) <= s1[1]
prob2.constraints += norm(s2[2:end]) <= s2[1]
solve!(prob2, ECOS.Optimizer)

G_ort = [G_ort_caps;G_ort_cone]
h_ort = [h_ort_caps;h_ort_cone]

G_soc1 = G_soc_caps
h_soc1 = h_soc_caps
G_soc2 = G_soc_cone
h_soc2 = h_soc_cone

G = [G_ort;G_soc1;G_soc2]
h = [h_ort;h_soc1;h_soc2]

n_ort = size(G_ort,1)
n_soc1 = size(G_soc1,1)
n_soc2 = size(G_soc2,1)

using JLD2
# jldsave("example_socp.jld2", G_ort=G_ort, h_ort=h_ort, G_soc=G_soc, h_soc=h_soc)
jldsave("example_socp_2.jld2", G = G , h=h , n_ort=n_ort, n_soc1 = n_soc1, n_soc2=n_soc2)

@info α
@show vec(z.value)
@show [x;α;γ]

@show norm(vec(z.value) - [x;α;γ])

# build big ones
build_primitive!(vis2, cone, :cone_int; α = α,color = mc.RGBA(0.1, 0.7, 0.7, 0.4))
update_pose!(vis2[:cone_int],cone)
build_primitive!(vis2, capsule, :capsule_int; α = α,color = mc.RGBA(0.7, 1.0, 0.7, 0.4))
update_pose!(vis2[:capsule_int],capsule)

spha = mc.HyperSphere(mc.Point(x...), 0.02)
mc.setobject!(vis2[:intersec], spha, mc.MeshPhongMaterial(color=mc.RGBA(1.0,0,0,1.0)))

# @show α

# c = cone.r - α*(cone.H/2)*bx # bottom of cone
# d = cone.r + α*(cone.H/2)*bx # top of cone
#
# sphc = mc.HyperSphere(mc.Point(c...), 0.02)
# mc.setobject!(vis2[:cone_c], sphc, mc.MeshPhongMaterial(color=mc.RGBA(0.0,1.0,0,1.0)))
# sphd = mc.HyperSphere(mc.Point(d...), 0.02)
# mc.setobject!(vis2[:cone_d], sphd, mc.MeshPhongMaterial(color=mc.RGBA(0.0,1.0,0,1.0)))
