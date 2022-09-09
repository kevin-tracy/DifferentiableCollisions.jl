using Pkg
Pkg.activate(dirname(@__DIR__))
using LinearAlgebra
import MeshCat as mc
import DCOL as dc
using StaticArrays

mutable struct Ellipsoid{T}
	# x'*Q*x ≦ 1.0
	r::SVector{3,T}
    q::SVector{4,T}
    Q::SMatrix{3,3,T,9}
	U::SMatrix{3,3,T,9}
	F::Eigen{T, T, SMatrix{3, 3, T, 9}, SVector{3, T}}
    function Ellipsoid(Q::SMatrix{3,3,T,9}) where {T}
        new{T}(
		SA[0,0,0.0],
		SA[1,0,0,0.0],
		Q,
		SMatrix{3,3}(cholesky(Hermitian(Q)).U),
		eigen(Q)
		)
    end
end
# function rand_ell(v::SVector{3,Float64})
# 	R = svd(@SMatrix randn(3,3)).U
# 	R*Diagonal(v)*R'
# end
function update_pose!(vis,P::Ellipsoid{T},name) where {T}
    mc.settransform!(vis[name], mc.Translation(P.r) ∘ mc.LinearMap(dc.dcm_from_q(P.q)*P.F.vectors))
end

function build_primitive!(vis,P::Ellipsoid{T},poly_name;color = mc.RGBA(0.7, 0.7, 0.7, 1.0), α = 1) where {T}
    e = mc.HyperEllipsoid(mc.Point(0,0,0.0), mc.Vec(α*(sqrt.(1 ./ P.F.values))))
    mc.setobject!(vis[poly_name], e, mc.MeshPhongMaterial(color = color))
    return nothing
end
# Q = (@SMatrix randn(3,3));Q = Q'*Q + I;

# E1 = Ellipsoid(rand_ell(SA[1,2,3.0]))
# E1 = Ellipsoid(inv(diagm(SA[1,2,3.0] .^ 2)))
vis = mc.Visualizer()
mc.open(vis)
Q = SMatrix{3,3}(Diagonal([1,1/2,1/3] .^ 2))
E1 = Ellipsoid(Q)
build_primitive!(vis,E1,:e1;α = 1.0, color = mc.RGBA(0.7, 0.7, 0.7, 0.1))
E1.q = [cosd(45/2);0;0;sind(45/2)]
update_pose!(vis,E1,:e1)

# for i = 1:1000
global iter = 0
for x = range(-2,2,length = 20)
	for y = range(-3,3,length = 20)
		for z = range(-4,4,length = 20)
			global iter += 1
			p = [x,y,z]
			if (p'*E1.Q*p) <= 1.0
				mc.setobject!(vis["spheree"*string(iter)],mc.HyperSphere(mc.Point(p...),0.05), mc.MeshPhongMaterial(color = mc.RGBA(0.0,1,0,1.0)))
			else
				# mc.setobject!(vis["spheree"*string(iter)],mc.HyperSphere(mc.Point(p...),0.05), mc.MeshPhongMaterial(color = mc.RGBA(1.0,0,0,1.0)))
			end
		end
	end
end



# spha = mc.HyperSphere(mc.Point(0,0,0.0), α*C.R)
# mc.setobject!(vis[sphere_name], spha, mc.MeshPhongMaterial(color=color))
# # F = eigen(Q)
# # S = F.vectors
# # D = Diagonal(F.values)
# e = mc.HyperEllipsoid(mc.Point(0,0,0.0), mc.Vec(E1.F.values...))
#
# mc.setobject!(vis[:e], e, mc.MeshPhongMaterial(color = mc.RGBA(1,0,0,.3)))
# mc.settransform!(vis[:e], mc.LinearMap(F.vectors * dc.dcm_from_mrp(SA[0,0,1]*tand(45/4))))
#
# cylx = mc.Cylinder(mc.Point(0,0,0.0), mc.Point((1.01*F.values[1]*S[:,1])...), 0.1)
# cyly = mc.Cylinder(mc.Point(0,0,0.0), mc.Point((1.01*F.values[2]*S[:,2])...), 0.1)
# cylz = mc.Cylinder(mc.Point(0,0,0.0), mc.Point((1.01*F.values[3]*S[:,3])...), 0.1)
#
# mc.setobject!(vis[:x], cylx, mc.MeshPhongMaterial(color = mc.RGBA(0,0,0,.3)))
# mc.setobject!(vis[:y], cyly, mc.MeshPhongMaterial(color = mc.RGBA(0,0,0,.3)))
# mc.setobject!(vis[:z], cylz, mc.MeshPhongMaterial(color = mc.RGBA(0,0,0,.3)))
