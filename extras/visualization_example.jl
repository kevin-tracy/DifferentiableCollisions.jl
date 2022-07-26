cd("/Users/kevintracy/.julia/dev/DCD/extras")
import Pkg; Pkg.activate(".")
import DCD
using BenchmarkTools
# import ForwardDiff as FD
# include("primitives.jl")
# using LinearAlgebra, StaticArrays, Convex, ECOS
using LinearAlgebra, StaticArrays
import MeshCat as mc
# collision between cone and capsule
vis1 = mc.Visualizer()
open(vis1)
vis2 = mc.Visualizer()
open(vis2)
let
    cone = DCD.Cone(2.0,deg2rad(22))
    cone.r = 0.3*(@SVector randn(3))
    cone.q = normalize((@SVector randn(4)))
    DCD.build_primitive!(vis1, cone, :cone; α = 1.0,color = mc.RGBA(0.1, 0.7, 0.7, 0.7))
    DCD.update_pose!(vis1[:cone],cone)

    capsule = DCD.Capsule(.3,1.2)
    capsule.r = (@SVector randn(3))
    capsule.q = normalize((@SVector randn(4)))
    DCD.build_primitive!(vis1, capsule, :capsule; α = 1.0,color = mc.RGBA(0.7, 1.0, 0.7, 0.7))
    DCD.update_pose!(vis1[:capsule],capsule)

    # α, x = DCD.proximity(capsule,cone)
    # @btime DCD.proximity($capsule,$cone)
    α, x = DCD.proximity(cone,capsule)
    @btime DCD.proximity($capsule,$cone)
    @info α

    α, x, ∂α_∂state = DCD.proximity_gradient(cone,capsule)

    @btime DCD.proximity_gradient($capsule,$cone)

    # build big ones
    DCD.build_primitive!(vis2, cone, :cone_int; α = α,color = mc.RGBA(0.1, 0.7, 0.7, 0.4))
    DCD.update_pose!(vis2[:cone_int],cone)
    DCD.build_primitive!(vis2, capsule, :capsule_int; α = α,color = mc.RGBA(0.7, 1.0, 0.7, 0.4))
    DCD.update_pose!(vis2[:capsule_int],capsule)

    spha = mc.HyperSphere(mc.Point(x...), 0.02)
    mc.setobject!(vis2[:intersec], spha, mc.MeshPhongMaterial(color=mc.RGBA(1.0,0,0,1.0)))

end
