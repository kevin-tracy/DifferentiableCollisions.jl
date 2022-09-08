using Pkg
Pkg.activate(dirname(@__DIR__))
using LinearAlgebra
import MeshCat as mc
import DCOL as dc
# vis = mc.Visualizer()
# mc.open(vis)



Q = randn(3,3);Q = Q'*Q + I;
F = eigen(Q)
S = F.vectors
D = Diagonal(F.values)
e = mc.HyperEllipsoid(mc.Point(0,0,0.0), mc.Vec(F.values...))

mc.setobject!(vis[:e], e, mc.MeshPhongMaterial(color = mc.RGBA(1,0,0,.3)))
mc.settransform!(vis[:e], mc.LinearMap(F.vectors * ))

cylx = mc.Cylinder(mc.Point(0,0,0.0), mc.Point((1.01*F.values[1]*S[:,1])...), 0.1)
cyly = mc.Cylinder(mc.Point(0,0,0.0), mc.Point((1.01*F.values[2]*S[:,2])...), 0.1)
cylz = mc.Cylinder(mc.Point(0,0,0.0), mc.Point((1.01*F.values[3]*S[:,3])...), 0.1)

mc.setobject!(vis[:x], cylx, mc.MeshPhongMaterial(color = mc.RGBA(0,0,0,.3)))
mc.setobject!(vis[:y], cyly, mc.MeshPhongMaterial(color = mc.RGBA(0,0,0,.3)))
mc.setobject!(vis[:z], cylz, mc.MeshPhongMaterial(color = mc.RGBA(0,0,0,.3)))
