cd("/Users/kevintracy/.julia/dev/DCD/extras")
import Pkg; Pkg.activate(".")
import DCD
using BenchmarkTools
import FiniteDiff
# import ForwardDiff as FD
# include("primitives.jl")
# using LinearAlgebra, StaticArrays, Convex, ECOS
using LinearAlgebra, StaticArrays
import MeshCat as mc
# collision between cone and capsule
# vis1 = mc.Visualizer()
# open(vis1)
# vis2 = mc.Visualizer()
# open(vis2)
let

    c1 = [245, 155, 66]/255
    c2 = [2,190,207]/255

    cone = DCD.Cone(2.0,deg2rad(22))
    cone.r = 0.3*(@SVector randn(3))
    cone.q = normalize((@SVector randn(4)))
    DCD.build_primitive!(vis1, cone, :cone; α = 1.0,color = mc.RGBA(c1..., 0.7))
    DCD.update_pose!(vis1[:cone],cone)

    capsule = DCD.Capsule(.3,1.2)
    capsule.r = (@SVector randn(3))
    capsule.q = normalize((@SVector randn(4)))
    DCD.build_primitive!(vis1, capsule, :capsule; α = 1.0,color = mc.RGBA(c2..., 0.7))
    DCD.update_pose!(vis1[:capsule],capsule)

    # α, x = DCD.proximity(capsule,cone)
    # @btime DCD.proximity($capsule,$cone)
    α, x = DCD.proximity(cone,capsule)
    # @btime DCD.proximity($capsule,$cone)
    @info α

    α, x, ∂α_∂state = DCD.proximity_gradient(cone,capsule)

    # @btime DCD.proximity_gradient($capsule,$cone)

    # build big ones
    DCD.build_primitive!(vis2, cone, :cone_int; α = α,color = mc.RGBA(c1..., 0.4))
    DCD.update_pose!(vis2[:cone_int],cone)
    DCD.build_primitive!(vis2, capsule, :capsule_int; α = α,color = mc.RGBA(c2..., 0.4))
    DCD.update_pose!(vis2[:capsule_int],capsule)

    # contact point on vis2
    spha = mc.HyperSphere(mc.Point(x...), 0.04)
    mc.setobject!(vis2[:intersec], spha, mc.MeshPhongMaterial(color=mc.RGBA(1.0,0,0,1.0)))

    pcone    = cone.r    + (x -    cone.r)/α
    pcapsule = capsule.r + (x - capsule.r)/α

    sph_pcone = mc.HyperSphere(mc.Point(pcone...), 0.04)
    mc.setobject!(vis1[:pcone], sph_pcone, mc.MeshPhongMaterial(color=mc.RGBA(1.0,0,0,1.0)))
    sph_pcaps = mc.HyperSphere(mc.Point(pcapsule...), 0.04)
    mc.setobject!(vis1[:pcaps], sph_pcaps, mc.MeshPhongMaterial(color=mc.RGBA(1.0,0,0,1.0)))

end
