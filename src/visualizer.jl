
function update_pose!(vis,P::AbstractPrimitive)
    mc.settransform!(vis, mc.Translation(P.r) ∘ mc.LinearMap(dcm_from_q(P.q)))
end
function update_pose!(vis,P::AbstractPrimitiveMRP)
    mc.settransform!(vis, mc.Translation(P.r) ∘ mc.LinearMap(dcm_from_mrp(P.p)))
end

function build_primitive!(vis,P::Union{Polytope{n,n3,T},PolytopeMRP{n,n3,T}},poly_name::Symbol;color = mc.RGBA(0.7, 0.7, 0.7, 1.0), α = 1) where {n,n3,T}
    N = length(P.b)
    h = Polyhedra.HalfSpace(P.A[1,:],α*P.b[1])
    for i = 2:N
        h = h ∩ Polyhedra.HalfSpace(P.A[i,:],α*P.b[i])
    end
    pol = Polyhedra.polyhedron(h)
    mc.setobject!(vis[poly_name], Polyhedra.Mesh(pol), mc.MeshPhongMaterial(color = color))
    return nothing
end

function build_primitive!(vis, C::Union{Capsule{T},CapsuleMRP{T}}, cap_name::Symbol;color = mc.RGBA(0.7, 0.7, 0.7, 1.0), α = 1) where {T}
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


function build_primitive!(vis, C::Union{Cylinder{T},CylinderMRP{T}}, cyl_name::Symbol;color = mc.RGBA(0.7, 0.7, 0.7, 1.0), α = 1) where {T}
    a = α*[-C.L/2,0,0]
    b = α*[C.L/2,0,0]
    cyl = mc.Cylinder(mc.Point(a...), mc.Point(b...), α*C.R)
    mc.setobject!(vis[cyl_name][:cyl], cyl, mc.MeshPhongMaterial(color=color))
    return nothing
end


function build_primitive!(vis, C::Union{Cone{T},ConeMRP{T}},cone_name::Symbol; color = mc.RGBA(0.7, 0.7, 0.7, 1.0), α = 1) where {T}
	W = tan(C.β)*C.H
	cc = mc.Cone(mc.Point(α*C.H/4,0,0), mc.Point(-α*C.H*3/4, 0.0, 0), α*W)
	mc.setobject!(vis[cone_name], cc, mc.MeshPhongMaterial(color = color))
end


function build_primitive!(vis, C::Union{Sphere{T},SphereMRP{T}},sphere_name::Symbol; color = mc.RGBA(0.7, 0.7, 0.7, 1.0), α = 1) where {T}
	spha = mc.HyperSphere(mc.Point(0,0,0.0), α*C.R)
	mc.setobject!(vis[sphere_name], spha, mc.MeshPhongMaterial(color=color))
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
    P = Polyhedra.polyhedron(h)

    # convert to vertices
    V = Polyhedra.vrep(P)
    verts = V.points.points

    # sort based on angle
    angs = [wrap_to_2pi(atan(v[2],v[1])) for v in verts]
    idx = sortperm(angs)
    return verts[idx] # return ordered vertices
end
function build_primitive!(vis,P::Union{Polygon{nh,nh2,T},PolygonMRP{nh,nh2,T}}, poly_name::Symbol; color = mc.RGBA(0.7, 0.7, 0.7, 1.0), α = 1 ) where {nh,nh2,T}

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
    pol = Polyhedra.polyhedron(h)
    mc.setobject!(vis[poly_name][:central], Polyhedra.Mesh(pol), mc.MeshPhongMaterial(color = color))
    return nothing
end

function add_axes!(vis,axes_name, scale, R; head_l = 0.1, head_w = 0.05)
	red_col =mc.RGBA(1.0,0.0,0.0,1.0)
	green_col =mc.RGBA(0.0,1.0,0.0,1.0)
	blue_col =mc.RGBA(0.0,0.0,1.0,1.0)

	cylx = mc.Cylinder(mc.Point(0,0,0.0), mc.Point(scale,0,0.0), R)
	cyly = mc.Cylinder(mc.Point(0,0,0.0), mc.Point(0,scale,0.0), R)
	cylz = mc.Cylinder(mc.Point(0,0,0.0), mc.Point(0,0.0,scale), R)

    mc.setobject!(vis[axes_name][:x], cylx, mc.MeshPhongMaterial(color=red_col))
	head = mc.Cone(mc.Point(scale,0,0.0), mc.Point(scale + head_l, 0.0, 0), head_w)
	mc.setobject!(vis[axes_name][:hx], head, mc.MeshPhongMaterial(color=red_col))

	mc.setobject!(vis[axes_name][:y], cyly, mc.MeshPhongMaterial(color=green_col))
	head = mc.Cone(mc.Point(0,scale,0.0), mc.Point(0, scale + head_l, 0.0), head_w)
	mc.setobject!(vis[axes_name][:hy], head, mc.MeshPhongMaterial(color=green_col))

	mc.setobject!(vis[axes_name][:z], cylz, mc.MeshPhongMaterial(color=blue_col))
	head = mc.Cone(mc.Point(0,0.0,scale), mc.Point(0, 0.0, scale + head_l), head_w)
	mc.setobject!(vis[axes_name][:hz], head, mc.MeshPhongMaterial(color=blue_col))
	return nothing
end

function axes_pair_to_quaternion(n1, n2)
	if norm(n1 + n2, Inf) < 1e-5
		n2 = n2 + 1e-5ones(3)
	end

	reg(x) = 1e-20 * (x == 0) + x
	# provides the quaternion that rotates n1 into n2, assuming n1 and n2 are normalized
	n1 ./= reg(norm(n1))
	n2 ./= reg(norm(n2))
	n3 = cross(n1,n2)
	cθ = n1' * n2 # cosine
	sθ = norm(n3) # sine
	axis = n3 ./ reg(sθ)
	tanθhalf = sθ / reg(1 + cθ)
	q = [1; tanθhalf * axis]
	q /= norm(q)
	return dcm_from_q(SVector{4}(q))
end
function set_floor!(vis;
	    x=20.0,
	    y=20.0,
	    z=0.1,
	    origin=[0,0,0.0],
		normal=[0,0,1.0],
	    color=mc.RGBA(0.5,0.5,0.5,1.0),
	    tilepermeter=0.5,
	    imagename="/Users/kevintracy/.julia/dev/DifferentialProximity/extras/polyhedra_plotting/tile.png",
	    axis::Bool=false,
	    grid::Bool=false)
    image = mc.PngImage(imagename)
    repeat = Int.(ceil.(tilepermeter * [x, y]))
    texture = mc.Texture(image=image, wrap=(1,1), repeat=(repeat[1],repeat[2]))
    mat = mc.MeshPhongMaterial(map=texture)
    # (color != nothing) && (mat = MeshPhongMaterial(color=color))
    obj = mc.HyperRectangle(mc.Vec(-x/2, -y/2, -z), mc.Vec(x, y, z))
    mc.setobject!(vis[:floor], obj, mat)
	p = origin
	q = axes_pair_to_quaternion([0,0,1.], normal)
    mc.settransform!(vis[:floor], mc.compose(
		mc.Translation(p...),
		mc.LinearMap(q),
		))

    # mc.setvisible!(vis["/Axes"], axis)
    # mc.setvisible!(vis["/Grid"], grid)
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
