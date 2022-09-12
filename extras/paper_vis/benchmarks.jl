cd("/Users/kevintracy/.julia/dev/DCOL/extras")
import Pkg; Pkg.activate(".")
import DCOL as DCD
using LinearAlgebra, StaticArrays
import MeshCat as mc
using JLD2
# using Convex
# import ECOS
using BenchmarkTools
import Random
using Colors
import Random
Random.seed!(1)

function create_n_sided(N,d)
    ns = [ [cos(θ);sin(θ)] for θ = 0:(2*π/N):(2*π*(N-1)/N)]
    A = vcat(transpose.((ns))...)
    b = d*ones(N)
    return SMatrix{N,2}(A), SVector{N}(b)
end
function create_rect_prism(;len = 20.0, wid = 20.0, hei = 2.0)
    ns = [SA[1,0,0.0], SA[0,1,0.0], SA[0,0,1.0],SA[-1,0,0.0], SA[0,-1,0.0], SA[0,0,-1.0]]
    cs = [SA[len/2,0,0.0], SA[0,wid/2,0.0], SA[0,0,hei/2],SA[-len/2,0,0.0], SA[0,-wid/2,0.0], SA[0,0,-hei/2]]

    A = zeros(6,3)
    b = zeros(6)

    for i = 1:6
        A[i,:] = ns[i]'
        b[i] = dot(ns[i],cs[i])
    end

    A = SMatrix{6,3}(A)
    b = SVector{6}(b)

    mass = len*wid*hei

    inertia = (mass/12)*Diagonal(SA[wid^2 + hei^2, len^2 + hei^2, len^2 + wid^2])

    return DCD.Polytope(A,b), mass, inertia
end

function run_bench()
    @load "/Users/kevintracy/.julia/dev/DifferentialProximity/extras/polyhedra_plotting/polytopes.jld2"
    b2 = 0.7*b2


    # import Random
    Random.seed!(20)

    Ps = [
    # DCD.Polytope(SMatrix{8,3}(A2),SVector{8}(b2))
    create_rect_prism(;len = 4.0, wid = 4.0, hei = 3.0)[1]
    DCD.Capsule(0.5,2.0)
    DCD.Cylinder(0.5,2.0)
    DCD.Cone(2.0,deg2rad(22))
    DCD.Sphere(0.7)
    DCD.Ellipsoid(SMatrix{3,3}(Diagonal((2.5*[1,1/2,1/3]) .^ 2)))
    DCD.Polygon(create_n_sided(5,0.95)...,0.1)
    ]
    for i = 1:length(Ps)
        Ps[i].q = normalize((@SVector randn(4)))
        Ps[i].r = 2*(@SVector randn(3))
    end

    btime_table = NaN*zeros(length(Ps),length(Ps))
    for i = 1:length(Ps)
        for j = 1:length(Ps)
            if i != j
                p1 = Ps[i]
                p2 = Ps[j]
                # @show typeof(p1)
                # @show typeof(p2)
                DCD.proximity(p1,p2;verbose = false)
                bresults = @benchmark DCD.proximity($p1,$p2)
                btime_table[i,j] = median(bresults.times)
            end
        end
    end
    @show btime_table
    println(btime_table)


    btime_table2 = NaN*zeros(length(Ps),length(Ps))
    for i = 1:length(Ps)
        for j = 1:length(Ps)
            if i != j
                p1 = Ps[i]
                p2 = Ps[j]
                # @show typeof(p1)
                # @show typeof(p2)
                DCD.proximity(p1,p2;verbose = false)
                bresults = @benchmark DCD.proximity_jacobian($p1,$p2)
                btime_table2[i,j] = median(bresults.times)
            end
        end
    end
    @show btime_table2
    println(btime_table2)
end

run_bench()
