using Pkg
Pkg.activate(dirname(@__DIR__))
using LinearAlgebra
import MeshCat as mc
import DCOL as dc
using StaticArrays
using Convex
using ECOS

# mutable struct Ellipsoid{T}
# 	# x'*Q*x ≦ 1.0
# 	r::SVector{3,T}
#     q::SVector{4,T}
#     P::SMatrix{3,3,T,9}
# 	U::SMatrix{3,3,T,9}
# 	F::Eigen{T, T, SMatrix{3, 3, T, 9}, SVector{3, T}}
#     function Ellipsoid(P::SMatrix{3,3,T,9}) where {T}
#         new{T}(
# 		SA[0,0,0.0],
# 		SA[1,0,0,0.0],
# 		P,
# 		SMatrix{3,3}(cholesky(Hermitian(P)).U),
# 		eigen(P)
# 		)
#     end
# end

# function update_pose!(vis,P::Ellipsoid{T},name) where {T}
#     mc.settransform!(vis[name], mc.Translation(P.r) ∘ mc.LinearMap(dc.dcm_from_q(P.q)*P.F.vectors))
# end
#
# function build_primitive!(vis,P::Ellipsoid{T},poly_name;color = mc.RGBA(0.7, 0.7, 0.7, 1.0), α = 1) where {T}
#     e = mc.HyperEllipsoid(mc.Point(0,0,0.0), mc.Vec(α*(sqrt.(1 ./ P.F.values))))
#     mc.setobject!(vis[poly_name], e, mc.MeshPhongMaterial(color = color))
#     return nothing
# end
# Q = (@SMatrix randn(3,3));Q = Q'*Q + I;

# E1 = Ellipsoid(rand_ell(SA[1,2,3.0]))
# E1 = Ellipsoid(inv(diagm(SA[1,2,3.0] .^ 2)))
vis = mc.Visualizer()
mc.open(vis)


E1 = dc.Ellipsoid(SMatrix{3,3}(Diagonal([1,1/2,1/3] .^ 2)))
E2 = dc.Ellipsoid(SMatrix{3,3}(Diagonal([1/2,1/1.2,1/2.4] .^ 2)))

c1 = [245, 155, 66]/255
c2 = [2,190,207]/255
dc.build_primitive!(vis,E1,:e1;α = 1.0, color = mc.RGBA(c1..., 1.0))
E1.q = normalize((@SVector randn(4)))
E1.r = 2*(@SVector randn(3))
dc.update_pose!(vis,E1,:e1)
dc.build_primitive!(vis,E2,:e2;α = 1.0, color = mc.RGBA(c2..., 1.0))
E2.q = normalize((@SVector randn(4)))
E2.r = 3*(@SVector randn(3))
dc.update_pose!(vis,E2,:e2)

x = Variable(3)
α = Variable()

Q1 = dc.dcm_from_q(E1.q)
Q2 = dc.dcm_from_q(E2.q)

prob = minimize(α)
prob.constraints += norm(E1.U*Q1'*(x - E1.r)) <= α
prob.constraints += norm(E2.U*Q2'*(x - E2.r)) <= α
solve!(prob, ECOS.Optimizer)

x = vec(x.value)
α = α.value
@info α
dc.build_primitive!(vis,E1,:e1_big;α = α, color = mc.RGBA(c1..., 0.4))
dc.update_pose!(vis,E1,:e1_big)
dc.build_primitive!(vis,E2,:e2_big;α = α, color = mc.RGBA(c2..., 0.4))
dc.update_pose!(vis,E2,:e2_big)

sph = mc.HyperSphere(mc.Point(x...),0.07)
mc.setobject!(vis[:x],sph,  mc.MeshPhongMaterial(color = mc.RGB(1,0,0)))
