
function update_pose!(vis,P)
    mc.settransform!(vis, mc.Translation(P.r) ∘ mc.LinearMap(mc.QuatRotation(P.q)))
end

abstract type AbstractPrimitive end

mutable struct Polytope{n,n3,T} <: AbstractPrimitive
    r::SVector{3,T}
    q::SVector{4,T}
    A::SMatrix{n,3,T,n3}
    b::SVector{n,T}
    function Polytope(A::SMatrix{n,3,T,n3}, b::SVector{n,T}) where{n,n3,T}
        new{n,n3,T}(SA[0,0,0.0],SA[1,0,0,0.0],A,b)
    end
end

function build_primitive!(vis,P::Polytope{n,n3,T},poly_name::Symbol;color = mc.RGBA(0.7, 0.7, 0.7, 1.0), α = 1) where {n,n3,T}
    N = length(P.b)
    h = Polyhedra.HalfSpace(P.A[1,:],α*P.b[1])
    for i = 2:N
        h = h ∩ Polyhedra.HalfSpace(P.A[i,:],α*P.b[i])
    end
    P = polyhedron(h)
    mc.setobject!(vis[poly_name], Polyhedra.Mesh(polyhedron(h)), mc.MeshPhongMaterial(color = color))
    return nothing
end

mutable struct Capsule{T} <: AbstractPrimitive
    r::SVector{3,T}
    q::SVector{4,T}
    R::T
    L::T
    function Capsule(R::T,L::T) where {T}
        new{T}(SA[0,0,0.0],SA[1,0,0,0.0],R,L)
    end
end

function build_primitive!(vis, C::Capsule{T}, cap_name::Symbol;color = mc.RGBA(0.7, 0.7, 0.7, 1.0), α = 1) where {T}
    a = α*[-C.L/2,0,0]
    b = α*[C.L/2,0,0]
    cyl = mc.Cylinder(mc.Point(a...), mc.Point(b...), α*C.R)
    spha = mc.HyperSphere(mc.Point(a...), α*C.R)
    sphb = mc.HyperSphere(mc.Point(b...), α*C.R)
    mc.setobject!(vis[cap_name][:cyl], cyl, mc.MeshPhongMaterial(color=color))
    mc.setobject!(vis[cap_name][:spha], spha, mc.MeshPhongMaterial(color=color))
    mc.setobject!(vis[cap_name][:sphb], sphb, mc.MeshPhongMaterial(color=color))
    return nothing
end

mutable struct Cylinder{T} <: AbstractPrimitive
    r::SVector{3,T}
    q::SVector{4,T}
    R::T
    L::T
    function Cylinder(R::T,L::T) where {T}
        new{T}(SA[0,0,0.0],SA[1,0,0,0.0],R,L)
    end
end

function build_primitive!(vis, C::Cylinder{T}, cyl_name::Symbol;color = mc.RGBA(0.7, 0.7, 0.7, 1.0), α = 1) where {T}
    a = α*[-C.L/2,0,0]
    b = α*[C.L/2,0,0]
    cyl = mc.Cylinder(mc.Point(a...), mc.Point(b...), α*C.R)
    mc.setobject!(vis[cyl_name][:cyl], cyl, mc.MeshPhongMaterial(color=color))
    return nothing
end


mutable struct Cone{T} <: AbstractPrimitive
    r::SVector{3,T}
    q::SVector{4,T}
    H::T
    β::T
    function Cone(H::T,β::T) where {T}
        new{T}(SA[0,0,0.0],SA[1,0,0,0.0],H,β)
    end
end
function build_primitive!(vis, C::Cone{T},cone_name::Symbol; color = mc.RGBA(0.7, 0.7, 0.7, 1.0), α = 1) where {T}
	W = tan(C.β)*C.H
	cc = mc.Cone(mc.Point(α*C.H/2,0,0), mc.Point(-α*C.H/2, 0.0, 0), α*W)
	mc.setobject!(vis[cone_name], cc, mc.MeshPhongMaterial(color = color))
end

mutable struct Sphere{T} <: AbstractPrimitive
	r::SVector{3,T}
    q::SVector{4,T} # not needed
    R::T
    function Sphere(R::T) where {T}
        new{T}(SA[0,0,0.0],SA[1,0,0,0.0],R)
    end
end
function build_primitive!(vis, C::Sphere{T},sphere_name::Symbol; color = mc.RGBA(0.7, 0.7, 0.7, 1.0), α = 1) where {T}
	spha = mc.HyperSphere(mc.Point(0,0,0.0), α*C.R)
	mc.setobject!(vis[sphere_name], spha, mc.MeshPhongMaterial(color=color))
end


mutable struct Polygon{nh,nh2,T} <: AbstractPrimitive
	r::SVector{3,T}         # position in world frame
    q::SVector{4,T}         # quat for attitude (ᴺqᴮ)
    A::SMatrix{nh,2,T,nh2}  # polygon description (Ax≦b)
    b::SVector{nh,T}        # polygon description (Ax≦b)
    R::T                    # "cushion" radius
    function Polygon(A::SMatrix{nh,2,T,nh2},b::SVector{nh,T},R::T) where {nh,nh2,T}
        new{nh,nh2,T}(SA[0,0,0.0],SA[1,0,0,0.0],A,b,R)
    end
end

function wrap_to_2pi(theta)
    if theta < 0.0
        theta = -2*pi*(abs(theta)/(2*pi)-floor(abs(theta/(2*pi)))) + 2*pi
    else
        theta = 2*pi*(abs(theta)/(2*pi)-floor(abs(theta/(2*pi))))
    end
    return theta
end
function find_verts(A,b)
    N = length(b) # number of halfspaces

    # create polyhedra object
    h = Polyhedra.HalfSpace(A[1,:],b[1])
    for i = 2:N
        h = h ∩ Polyhedra.HalfSpace(A[i,:],b[i])
    end
    P = polyhedron(h)

    # convert to vertices
    V = vrep(P)
    verts = V.points.points

    # sort based on angle
    angs = [wrap_to_2pi(atan(v[2],v[1])) for v in verts]
    idx = sortperm(angs)
    return verts[idx] # return ordered vertices
end
function build_primitive!(vis,P::Polygon{nh,nh2,T}, poly_name::Symbol; color = mc.RGBA(0.7, 0.7, 0.7, 1.0), α = 1 ) where {nh,nh2,T}

	A, b, R = P.A, P.b, P.R
	b = α*b
	R = α*R
	N = length(b)

    # create 2d
    verts = find_verts(A,b)

    # corners
    for i = 1:length(verts)
        v = verts[i]
        p = mc.HyperSphere(mc.Point(v[1],v[2],0.0),R)
        mc.setobject!(vis[poly_name]["vert"*string(i)],p, mc.MeshPhongMaterial(color = color))
    end

    # cylinders
    for i = 1:length(verts)
        if i < length(verts)
            v1 = verts[i]
            v2 = verts[i+1]
            p1 = mc.Point(v1[1],v1[2],0.0)
            p2 = mc.Point(v2[1],v2[2],0.0)
            cyl = mc.Cylinder(p1, p2, R)
            mc.setobject!(vis[poly_name]["side"*string(i)],cyl, mc.MeshPhongMaterial(color = color))
        else
            v1 = verts[i]
            v2 = verts[1]
            p1 = mc.Point(v1[1],v1[2],0.0)
            p2 = mc.Point(v2[1],v2[2],0.0)
            cyl = mc.Cylinder(p1, p2, R)
            mc.setobject!(vis[poly_name]["side"*string(i)],cyl, mc.MeshPhongMaterial(color = color))
        end
    end

    # create polyhedra object
    h = Polyhedra.HalfSpace([A[1,:];0],b[1])
    for i = 2:N
        h = h ∩ Polyhedra.HalfSpace([A[i,:];0],b[i])
    end
    h = h ∩ Polyhedra.HalfSpace(SA[0,0,1],R) ∩ Polyhedra.HalfSpace(SA[0,0,-1],R)
    P = polyhedron(h)
    mc.setobject!(vis[poly_name][:central], Polyhedra.Mesh(polyhedron(h)), mc.MeshPhongMaterial(color = color))
    return nothing
end


# function create_n_sided(N,d)
#     ns = [ [cos(θ);sin(θ)] for θ = 0:(2*π/N):(2*π*(N-1)/N)]
#     A = vcat(transpose.((ns))...)
#     b = d*ones(N)
#     return SMatrix{N,2}(A), SVector{N}(b)
# end
# vis = mc.Visualizer()
# open(vis)
# let
#
#     # vis = Visualizer()
#
#     @load "polytopes.jld2"
#     polytope = Polytope(SMatrix{8,3}(A2),SVector{8}(0.5*b2))
#     polytope.r = SA[-7,0,0.0]
#     build_primitive!(vis,polytope,:polytope; α = 0.5)
#     update_pose!(vis[:polytope],polytope)
#
#     capsule = Capsule(.3,1.0)
#     capsule.r = SA[-5,0,0.0]
#     build_primitive!(vis,capsule,:capsule; α = 0.5)
#     update_pose!(vis[:capsule],capsule)
#
#     cylinder = Cylinder(.3,1.0)
#     cylinder.r = SA[-3,0,0.0]
#     build_primitive!(vis,cylinder,:cylinder; α = 0.5)
#     update_pose!(vis[:cylinder],cylinder)
#
# 	cone = Cone(2.0,deg2rad(25))
# 	cone.r = SA[-1,0,0.0]
# 	cone.q = SA[cos(pi/4),0,sin(pi/4),0]
# 	build_primitive!(vis, cone, :cone; α = 0.5)
# 	update_pose!(vis[:cone],cone)
#
# 	sphere = Sphere(0.7)
# 	sphere.r = SA[1,0,0]
# 	build_primitive!(vis,sphere,:sphere; α = 0.5)
# 	update_pose!(vis[:sphere], sphere)
#
# 	polygon = Polygon(create_n_sided(5,0.8)..., 0.2)
# 	polygon.r = SA[3,0,0]
# 	build_primitive!(vis,polygon,:polygon; α = 0.5)
# 	update_pose!(vis[:polygon], polygon)
#
# end
