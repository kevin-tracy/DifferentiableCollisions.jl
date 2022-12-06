@inline function mrp_from_q(q::SVector{4,T}) where {T}
    return q[SA[2,3,4]]/(1+q[1])
end

@inline function q_from_mrp(p::SVector{3,T}) where {T}
    return (1/(1+dot(p,p)))*vcat((1-dot(p,p)),2*p)
end

@inline function dcm_from_q(q::SVector{4,T}) where {T}
    q4,q1,q2,q3 = normalize(q)

    # DCM
    Q = @SArray [(2*q1^2+2*q4^2-1)   2*(q1*q2 - q3*q4)   2*(q1*q3 + q2*q4);
          2*(q1*q2 + q3*q4)  (2*q2^2+2*q4^2-1)   2*(q2*q3 - q1*q4);
          2*(q1*q3 - q2*q4)   2*(q2*q3 + q1*q4)  (2*q3^2+2*q4^2-1)]
end

@inline function dcm_from_mrp(p::SVector{3,T}) where {T}
    p1,p2,p3 = p
    den = (p1^2 + p2^2 + p3^2 + 1)^2
    a = (4*p1^2 + 4*p2^2 + 4*p3^2 - 4)
    SA[
    (-((8*p2^2+8*p3^2)/den-1)*den)   (8*p1*p2 + p3*a)     (8*p1*p3 - p2*a);
    (8*p1*p2 - p3*a) (-((8*p1^2 + 8*p3^2)/den - 1)*den)   (8*p2*p3 + p1*a);
    (8*p1*p3 + p2*a)  (8*p2*p3 - p1*a)  (-((8*p1^2 + 8*p2^2)/den - 1)*den)
    ]/den
end

# CAPSULE
@inline function capsule_problem_matrices(R::T,L::T,r::SVector{3,T1},n_Q_b::SMatrix{3,3,T2,9}) where {T,T1,T2}
    bx = n_Q_b*SA[1,0,0.0]
    G_soc_top = SA[0 0 0.0 -R 0]
    G_soc_bot = hcat(-Diagonal(SA[1,1,1.0]), SA[0,0,0.0], bx )
    G_soc = [G_soc_top;G_soc_bot]
    h_soc = [0; -(r)]
    G_ort = SA[0 0 0 -L/2 1; 0 0 0 -L/2 -1.0]
    h_ort = SA[0,0.0]
    G_ort, h_ort, G_soc, h_soc
end
@inline function problem_matrices(capsule::Capsule{T},r::SVector{3,T1},q::SVector{4,T2}) where {T,T1,T2}
    n_Q_b = dcm_from_q(q)
    capsule_problem_matrices(capsule.R,
                             capsule.L,
                             r + n_Q_b*capsule.r_offset,
                             n_Q_b*capsule.Q_offset)
end
@inline function problem_matrices(capsule::CapsuleMRP{T},r::SVector{3,T1},p::SVector{3,T2}) where {T,T1,T2}
    n_Q_b = dcm_from_mrp(p)
    capsule_problem_matrices(capsule.R,
                             capsule.L,
                             r + n_Q_b*capsule.r_offset,
                             n_Q_b*capsule.Q_offset)
end

# CONE
@inline function cone_problem_matrices(H::T,β::T,r::SVector{3,T1},n_Q_b::SMatrix{3,3,T2,9}) where {T,T1,T2}
    tanβ = tan(β)
    E = Diagonal(SA[tanβ,1,1.0])
    bx = n_Q_b*SA[1,0,0]
    EQt = E*n_Q_b'
    h_soc = -EQt*r
    G_soc = [(-EQt) (-SA[tanβ*3*H/4,0,0])]
    G_ort = SA[bx[1] bx[2] bx[3] -H/4]
    h_ort = SA[dot(bx,r)]
    G_ort, h_ort, G_soc, h_soc
end
@inline function problem_matrices(cone::Cone{T},r::SVector{3,T1},q::SVector{4,T2}) where {T,T1,T2}
    n_Q_b = dcm_from_q(q)
    cone_problem_matrices(cone.H,cone.β,r + n_Q_b*cone.r_offset, n_Q_b*cone.Q_offset)
end
@inline function problem_matrices(cone::ConeMRP{T},r::SVector{3,T1},p::SVector{3,T2}) where {T,T1,T2}
    n_Q_b = dcm_from_mrp(p)
    cone_problem_matrices(cone.H,cone.β,r + n_Q_b*cone.r_offset, n_Q_b*cone.Q_offset)
end



# polytope
@inline function polytope_problem_matrices(A::SMatrix{nh,3,T1,nh3},b::SVector{nh,T2},r::SVector{3,T3},n_Q_b::SMatrix{3,3,T4,9}) where {nh,nh3,T1,T2,T3,T4}
    AQt = A*n_Q_b'
    G_ort = [AQt  -b]
    h_ort = AQt*r

    h_soc = SArray{Tuple{0}, Float64, 1, 0}(())
    G_soc = SArray{Tuple{0, 4}, Float64, 2, 0}(())

    G_ort, h_ort, G_soc, h_soc
end
@inline function problem_matrices(polytope::Polytope{n,n3,T},r::SVector{3,T1},q::SVector{4,T2}) where {n,n3,T,T1,T2}
    n_Q_b = dcm_from_q(q)
    polytope_problem_matrices(polytope.A,polytope.b,r + n_Q_b*polytope.r_offset, n_Q_b*polytope.Q_offset)
end
@inline function problem_matrices(polytope::PolytopeMRP{n,n3,T},r::SVector{3,T1},p::SVector{3,T2}) where {n,n3,T,T1,T2}
    n_Q_b = dcm_from_mrp(p)
    polytope_problem_matrices(polytope.A,polytope.b,r + n_Q_b*polytope.r_offset, n_Q_b*polytope.Q_offset)
end

# sphere
@inline function sphere_problem_matrices(R::T,r::SVector{3,T1}) where {T,T1}
    h_ort = SArray{Tuple{0}, Float64, 1, 0}(())
    G_ort = SArray{Tuple{0, 4}, Float64, 2, 0}(())

    h_soc = [0;-r]

    G_soc = SA[
                 0  0  0 -R
                -1  0  0  0;
                 0 -1  0  0;
                 0  0 -1  0
                 ]

    return G_ort, h_ort, G_soc, h_soc
end
@inline function problem_matrices(sphere::Sphere{T},r::SVector{3,T1},q::SVector{4,T2}) where {T,T1,T2}
    n_Q_b = dcm_from_q(q)
    sphere_problem_matrices(sphere.R,r + n_Q_b*sphere.r_offset)
end
@inline function problem_matrices(sphere::SphereMRP{T},r::SVector{3,T1},p::SVector{3,T2}) where {T,T1,T2}
    n_Q_b = dcm_from_mrp(p)
    sphere_problem_matrices(sphere.R,r + n_Q_b*sphere.r_offset)
end

# padded polygon
@inline function polygon_problem_matrices(A::SMatrix{nh,2,T1,nh3},b::SVector{nh,T2},R::Float64, r::SVector{3,T3},n_Q_b::SMatrix{3,3,T4,9}) where {nh,nh3,T1,T2,T3,T4}
    Q̃ = n_Q_b[:,SA[1,2]]
    G_ort = hcat((@SMatrix zeros(nh,3)), -b, A)
    h_ort = @SVector zeros(nh)

    G_soc_top = SA[0 0 0 -R 0 0]
    G_soc_bot = hcat(SA[-1 0 0 0;0 -1 0 0;0 0 -1 0], Q̃)
    G_soc = [G_soc_top;G_soc_bot]
    h_soc = [0;-r]
    G_ort, h_ort, G_soc, h_soc
end
@inline function problem_matrices(polygon::Polygon{n,n3,T},r::SVector{3,T1},q::SVector{4,T2}) where {n,n3,T,T1,T2}
    n_Q_b = dcm_from_q(q)
    polygon_problem_matrices(polygon.A,polygon.b,polygon.R,r + n_Q_b*polygon.r_offset, n_Q_b*polygon.Q_offset)
end
@inline function problem_matrices(polygon::PolygonMRP{n,n3,T},r::SVector{3,T1},p::SVector{3,T2}) where {n,n3,T,T1,T2}
    n_Q_b = dcm_from_mrp(p)
    polygon_problem_matrices(polygon.A,polygon.b,polygon.R,r + n_Q_b*polygon.r_offset, n_Q_b*polygon.Q_offset)
end


# cylinder
@inline function cylinder_problem_matrices(R::T,L::T,r::SVector{3,T1},n_Q_b::SMatrix{3,3,T2,9}) where {T,T1,T2}
    bx = n_Q_b*SA[1,0,0.0]
    G_soc_top = SA[0 0 0.0 -R 0]
    G_soc_bot = hcat(-Diagonal(SA[1,1,1.0]), SA[0,0,0.0], bx )
    G_soc = [G_soc_top;G_soc_bot]
    h_soc = [0; -(r)]
    G_ort = SA[
                0 0 0 -L/2 1;
                0 0 0 -L/2 -1.0;
                -bx[1] -bx[2] -bx[3] -L/2 0;
                 bx[1]  bx[2]  bx[3] -L/2 0
            ]
    h_ort = SA[0,0.0, -dot(bx,r), dot(bx,r)]
    G_ort, h_ort, G_soc, h_soc
end
@inline function problem_matrices(cylinder::Cylinder{T},r::SVector{3,T1},q::SVector{4,T2}) where {T,T1,T2}
    n_Q_b = dcm_from_q(q)
    cylinder_problem_matrices(cylinder.R,cylinder.L,r + n_Q_b*cylinder.r_offset, n_Q_b*cylinder.Q_offset)
end
@inline function problem_matrices(cylinder::CylinderMRP{T},r::SVector{3,T1},p::SVector{3,T2}) where {T,T1,T2}
    n_Q_b = dcm_from_mrp(p)
    cylinder_problem_matrices(cylinder.R,cylinder.L,r + n_Q_b*cylinder.r_offset, n_Q_b*cylinder.Q_offset)
end


# Ellipsoid
@inline function ellipsoid_problem_matrices(U::SMatrix{3,3,T1,9},r::SVector{3,T2},Q::SMatrix{3,3,T3,9}) where {T1,T2,T3}
    h_ort = SArray{Tuple{0}, Float64, 1, 0}(())
    G_ort = SArray{Tuple{0, 4}, Float64, 2, 0}(())

    h_soc = [0;-U*Q'*r]

    G_soc_top = SA[0 0 0 -1]
    G_soc_bot = hcat(-U*Q', (@SVector zeros(3)))
    G_soc = [G_soc_top;G_soc_bot]
    G_ort, h_ort, G_soc, h_soc
end

@inline function problem_matrices(ellipsoid::Ellipsoid{T},r::SVector{3,T1},q::SVector{4,T2}) where {T,T1,T2}
    n_Q_b = dcm_from_q(q)
    ellipsoid_problem_matrices(ellipsoid.U, r + n_Q_b*ellipsoid.r_offset, n_Q_b*ellipsoid.Q_offset)
end
@inline function problem_matrices(ellipsoid::EllipsoidMRP{T},r::SVector{3,T1},p::SVector{3,T2}) where {T,T1,T2}
    n_Q_b = dcm_from_mrp(p)
    ellipsoid_problem_matrices(ellipsoid.U, r + n_Q_b*ellipsoid.r_offset, n_Q_b*ellipsoid.Q_offset)
end
