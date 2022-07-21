import MeshCat as mc
using Meshing
import GeometryTypes as gt
vis = mc.Visualizer()
open(vis)
f = x -> sum(sin, 5 * x)
# sdf = GeometryTypes.SignedDistanceField(f, mc.HyperRectangle(mc.Vec(-1, -1, -1), mc.Vec(2, 2, 2)))
s2 = gt.SignedDistanceField(gt.HyperRectangle(gt.Vec(0,0,0),gt.Vec(1,1,1))) do v
        sqrt(sum(v.*v)) - 1 # sphere
end

mesh = gt.HomogenousMesh(s2, Meshing.MarchingTetrahedra())
mc.setobject!(vis, mesh, mc.MeshPhongMaterial(color=mc.RGBA{Float32}(1, 0, 0, 0.5)))


           # s2 = GeometryTypes.SignedDistanceField(GeometryTypes.HyperRectangle(GeometryTypes.Vec(0,0,0),GeometryTypes.Vec(1,1,1))) do v
           #         sqrt(sum(v.*v)) - 1 # sphere
           # end
           #
