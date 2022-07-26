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

    pdip_tol = 1e-4

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
    α, x = DCD.proximity(cone,capsule; pdip_tol)
    # @btime DCD.proximity($capsule,$cone)
    @info α

    α, x, ∂z_∂state = DCD.proximity_jacobian(cone,capsule;pdip_tol)
    α2, x2, ∂z_∂state2 = DCD.proximity_jacobian_slow(cone,capsule;pdip_tol)

    # @btime DCD.proximity_jacobian($capsule,$cone)

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

    # check derivatives
    function fd_α(cone,capsule,r1,q1,r2,q2)
        cone.r = r1
        cone.q = q1
        capsule.r = r2
        capsule.q = q2
        α, x = DCD.proximity(cone,capsule; pdip_tol = 1e-12)
        [x;α]
    end

    idx_r1 = SVector{3}(1:3)
    idx_q1 = SVector{4}(4:7)
    idx_r2 = SVector{3}(8:10)
    idx_q2 = SVector{4}(11:14)

    J1 = FiniteDiff.finite_difference_jacobian(θ -> fd_α(cone,capsule,θ[idx_r1],θ[idx_q1],θ[idx_r2],θ[idx_q2]), [cone.r;cone.q;capsule.r;capsule.q])

    # @show J1
    # @show ∂z_∂state

    @show norm(J1 - ∂z_∂state)

    for i = 1:4
        @show norm(J1[i,:] - ∂z_∂state[i,:])
    end

    @show norm(J1 - ∂z_∂state2)
    @show norm(∂z_∂state - ∂z_∂state2)
    @show abs.(J1[1,:] -  ∂z_∂state[1,:])

    @show J1[1,:]
    @show ∂z_∂state[1,:]






end
