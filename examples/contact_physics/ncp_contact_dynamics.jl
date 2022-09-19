using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), ".."))
using DCOL
Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()

using LinearAlgebra
using Printf
using StaticArrays
import ForwardDiff as FD
import MeshCat as mc
import DCOL as dc
import Random
using Colors
using SparseArrays
using Combinatorics
using JLD2

include(joinpath(@__DIR__,"variational_utils.jl"))

function trans_part(m,x1,x2,x3,Δt)
    # translational DEL (add gravity here if you want)
    (1/Δt)*m*(x2-x1) - (1/Δt)*m*(x3-x2) #+  Δt*m*gravity
end

function rot_part(J,q1,q2,q3,Δt)
    # rotational DEL
    (2.0/Δt)*G(q2)'*L(q1)*H*J*H'*L(q1)'*q2 + (2.0/Δt)*G(q2)'*T*R(q3)'*H*J*H'*L(q2)'*q3
end

function single_DEL(z₋,z,z₊,J,m,h)
    # translational and rotational DEL together
    p₋ = z₋[1:3]
    q₋ = z₋[4:7]
    p = z[1:3]
    q = z[4:7]
    p₊ = z₊[1:3]
    q₊ = z₊[4:7]
    [
    trans_part(m,p₋,p,p₊,h);
    rot_part(J,q₋,q,q₊,h)
    ]
end
function create_indexing(N_bodies)
    # here we create indexing for an arbitrary number of bodies (all in DCOL)
    idx_z  = [((i - 1)*7 .+ (1:7)) for i = 1:N_bodies]
    idx_Δz = [((i - 1)*6 .+ (1:6)) for i = 1:N_bodies]

    interactions = collect(combinations(1:N_bodies,2))
    N_interactions = length(interactions)
    @assert N_interactions == binomial(N_bodies,2)

    idx_s = idx_z[end][end] .+ (1:N_interactions)
    idx_Δs = idx_s .- N_bodies
    idx_λ = idx_s[end][end] .+ (1:N_interactions)
    idx_Δλ = idx_λ .- N_bodies

    idx_α = 6*N_bodies .+ (1:N_interactions)

    # throw all the stuff we want in this idx named tuple
    idx = (
    z = idx_z,
    s = idx_s,
    λ = idx_λ,
    Δz = idx_Δz,
    Δs = idx_Δs,
    Δλ = idx_Δλ,
    interactions = interactions,
    N_interactions = N_interactions,
    N_bodies = N_bodies,
    α = idx_α
    )

    return idx
end


function update_objects!(P, w, idx)
    # update position and orientation of each DCOL struct
    for i = 1:idx.N_bodies
        P[i].r = w[idx.z[i][SA[1,2,3]]]
        P[i].q = normalize(w[idx.z[i][SA[4,5,6,7]]])
    end
    nothing
end

function contacts(P,idx)
    # contacts between every pair of DCOL bodies
    [(dc.proximity(P[i],P[j])[1] - 1) for (i,j) in idx.interactions]
end
function Gbar_2_body(z,idx_z1, idx_z2)
    # attitude jacobian for two rigid bodies
    Gbar1 = blockdiag(sparse(I(3)),sparse(G(z[idx_z1[4:7]])))
    Gbar2 = blockdiag(sparse(I(3)),sparse(G(z[idx_z2[4:7]])))
    Gbar = Matrix(blockdiag(Gbar1,Gbar2))
end
function Gbar(w,idx)
    # attitude jacobian for idx.interactions rigid bodies
    G1 = (blockdiag([blockdiag(sparse(I(3)),sparse(G(w[idx.z[i][4:7]]))) for i = 1:idx.N_bodies]...))
    Matrix(blockdiag(G1,sparse(I(2*length(idx.interactions)))))
end

function update_se3(state, delta)
    # update position additively, attitude multiplicatively
    [
    state[SA[1,2,3]] + delta[SA[1,2,3]];
    L(state[SA[4,5,6,7]]) * ρ(delta[SA[4,5,6]])
    ]
end
function update_w(w,Δ,idx)
    # update our variable w that has positions, attitudes, slacks and λ's
    wnew = deepcopy(w)
    for i = 1:idx.N_bodies
        wnew[idx.z[i]] = update_se3(w[idx.z[i]], Δ[idx.Δz[i]])
    end
    wnew[idx.s] += Δ[idx.Δs]
    wnew[idx.λ] += Δ[idx.Δλ]
    wnew
end

# modify the torque to account for the extra 2x from quaternion stuff
const τ_mod = Diagonal(kron(ones(2),[ones(3);0.5*ones(3)]))

function contact_kkt(w₋, w, w₊, P, inertias, masses, idx, κ)
    # KKT conditions for our NCP
    s = w₊[idx.s]
    λ = w₊[idx.λ]

    # DEL stuff for each body
    DELs = [single_DEL(w₋[idx.z[i]],w[idx.z[i]],w₊[idx.z[i]],inertias[i],masses[i],h) for i = 1:idx.N_bodies]

    # now we add the contact jacobian functions stuff at middle time step
    update_objects!(P,w,idx)
    for k = 1:idx.N_interactions
        i,j = idx.interactions[k]
        _,_,D_state = dc.proximity_jacobian(P[i],P[j]; pdip_tol = 1e-6)
        D_α = reshape(D_state[4,:],1,14)
        E = h * τ_mod * (D_α*Gbar_2_body(w, idx.z[i], idx.z[j]))'*[λ[k]]
        DELs[i] += E[1:6]
        DELs[j] += E[7:12]
    end

    # now get contact stuff for + time step
    update_objects!(P,w₊,idx)
    αs = contacts(P,idx)

    [
    vcat(DELs...);  # DEL's + contact jacobians
    s - αs;         # slack (s) must equal contact αs
    (s .* λ) .- κ;  # complimentarity between s and λ
    ]
end
function contact_kkt_no_α(w₋, w, w₊, P, inertias, masses, idx, κ)
    # here is the same function as above, but without any contact stuff for
    # the + time step, this allows us to forwarddiff this function, and add
    # our DCOL contact jacobians seperately

    s = w₊[idx.s]
    λ = w₊[idx.λ]

    # DEL stuff for each body
    DELs = [single_DEL(w₋[idx.z[i]],w[idx.z[i]],w₊[idx.z[i]],inertias[i],masses[i],h) for i = 1:length(P)]

    # now we add the jacobian functions stuff at middle time step
    update_objects!(P,w,idx)
    for k = 1:idx.N_interactions
        i,j = idx.interactions[k]
        _,_,D_state = dc.proximity_jacobian(P[i],P[j]; pdip_tol = 1e-6)
        D_α = reshape(D_state[4,:],1,14)
        E = h * τ_mod * (D_α*Gbar_2_body(w, idx.z[i], idx.z[j]))'*[λ[k]]
        DELs[i] += E[1:6]
        DELs[j] += E[7:12]
    end

    # this is commented out (see above function)
    # now get contact stuff for + time step
    # update_objects!(P,w₊,idx)
    # αs = contacts(P,idx)

    [
    vcat(DELs...);
    s; #- αs;
    (s .* λ) .- κ;
    ]
end
function linesearch(x,dx)
    # nonnegative orthant analytical linesearch (check cvxopt documentation)
    α = min(0.99, minimum([dx[i]<0 ? -x[i]/dx[i] : Inf for i = 1:length(x)]))
end
function ncp_solve(w₋, w, P, inertias, masses, idx)
    w₊ = copy(w) #+ .1*abs.(randn(length(z)))
    w₊[idx.s] .+= 1
    w₊[idx.λ] .+= 1

    @printf "iter    |∇ₓL|      |c|        κ          μ          α         αs        αλ\n"
    @printf "--------------------------------------------------------------------------\n"

    for main_iter = 1:30
        rhs1 = -contact_kkt(w₋, w, w₊, P, inertias, masses, idx, 0)
        if norm(rhs1)<1e-6
            @info "success"
            return w₊
        end

        # forward diff for contact KKT
        D = FD.jacobian(_w -> contact_kkt_no_α(w₋, w, _w, P, inertias, masses, idx, 0.0),w₊)

        # add DCOL contact jacobians in for all our stuff
        update_objects!(P,w₊,idx)
        for k = 1:idx.N_interactions
            i,j = idx.interactions[k]
            (dc.proximity(P[i],P[j])[1] - 1)
            _,_,D_α = dc.proximity_jacobian(P[i],P[j])
            D_α = reshape(D_α[4,:],1,14)
            D[idx.α[k],idx.z[i]] = -D_α[1,idx.z[1]]
            D[idx.α[k],idx.z[j]] = -D_α[1,idx.z[2]]
        end


        # use G bar to handle quats, then factorize the matrix
        F = factorize(D*Gbar(w₊,idx))

        # affine step
        Δa = F\rhs1
        αa = 0.99*min(linesearch(w₊[idx.s], Δa[idx.Δs]), linesearch(w₊[idx.λ], Δa[idx.Δλ]))

        # duality gap growth
        μ = dot(w₊[idx.s], w₊[idx.λ])
        μa = dot(w₊[idx.s] + αa*Δa[idx.Δs], w₊[idx.λ] + αa*Δa[idx.Δλ])
        σ = min(0.99,max(0,μa/μ))^3
        κ = max(min(σ*μ,1),1e-8)

        # centering step
        rhs2 = -contact_kkt(w₋, w, w₊, P, inertias, masses, idx, κ)
        Δ = F\rhs2

        # linesearch
        α1 = linesearch(w₊[idx.s], Δ[idx.Δs])
        α2 = linesearch(w₊[idx.λ], Δ[idx.Δλ])
        α = 0.99*min(α1, α2)

        # update
        w₊ = update_w(w₊,α*Δ,idx)

        @printf("%3d    %9.2e  %9.2e  %9.2e  %9.2e  % 6.4f % 6.4f % 6.4f\n",
          main_iter, norm(rhs1[1:(6*idx.N_bodies)]), norm(rhs1[(6*idx.N_bodies) .+ (1:idx.N_interactions)]), κ, μ, α, α1, α2)

    end
    error("NCP solver failed")
end


# time step
const h = 0.05

let

    Random.seed!(1)


    path_str = joinpath(dirname(dirname(@__DIR__)),"test/example_socps/polytopes.jld2")
    f = jldopen(path_str)
    A1 = SMatrix{14,3}(f["A1"])
    b1 = SVector{14}(f["b1"])
    A2 = SMatrix{8,3}(f["A2"])
    b2 = SVector{8}(f["b2"])

    # build up all of the objects in your scene
    P = [dc.Cylinder(0.6,3.0), dc.Capsule(0.2,5.0), dc.Sphere(0.8),
         dc.Cone(2.0, deg2rad(22)),dc.Polytope(SMatrix{8,3}(A2),SVector{8}(b2)),dc.Polygon(dc.create_n_sided(5,0.6)...,0.2),
         dc.Cylinder(1.1,2.3), dc.Capsule(0.8,1.0), dc.Sphere(0.5),
              dc.Cone(3.0, deg2rad(18)),dc.Polytope(SMatrix{14,3}(A1),SVector{14}(b1)),dc.Polygon(dc.create_n_sided(8,0.8)...,0.15)]
    P_floor, mass_floor, inertia_floor = dc.create_rect_prism(1.0,1.0,1.0;attitude = :quat)
    push!(P,P_floor)

    # create the indexing named tuple
    N_bodies = length(P)
    @assert length(P) == N_bodies
    idx = create_indexing(N_bodies)

    # choose masses and inertias for everything (this is done lazily here)
    masses = ones(N_bodies)
    inertias = [Diagonal(SA[1,2,3.0]) for i = 1:N_bodies]

    # initial conditions of everything
    rs = [5*(@SVector randn(3)) for i = 1:N_bodies]
    rs[end] = SA[0,0,0.0]
    qs = [SA[1,0,0,0.0] for i = 1:N_bodies]

    # put it all in a vector w (this is the full vector for the ncp solve)
    w0 = vcat([[rs[i];qs[i]] for i = 1:N_bodies]..., zeros(2*idx.N_interactions))

    # initial velocities
    vs =  [SA[1,1,1.0] for i = 1:N_bodies]
    for i = 1:N_bodies
        vs[i] = -.5*rs[i]
    end
    ωs = [deg2rad.(20*(@SVector randn(3))) for i = 1:N_bodies]

    # use initial velocities to get a second initial condition
    r2s = [(rs[i] + h*vs[i]) for i = 1:N_bodies]
    q2s = [(L(qs[i])*Expq(h*ωs[i])) for i = 1:N_bodies]
    w1 = vcat([[r2s[i];q2s[i]] for i = 1:N_bodies]..., zeros(2*idx.N_interactions))

    # setup sim time
    N = 100
    W = [zeros(length(w0)) for i = 1:N]
    W[1] = w0
    W[2] = w1

    for i = 2:N-1
        println("------------------ITER NUMBER $i--------------------")
        W[i+1] = ncp_solve(W[i-1], W[i], P, inertias, masses, idx)
    end

    vis = mc.Visualizer()
    mc.open(vis)


    coll = Random.shuffle(range(HSVA(0,0.5,.9,1.0), stop=HSVA(-340,0.5,.9,1.0), length=N_bodies))
    for i = 1:N_bodies
        dc.build_primitive!(vis, P[i], Symbol("P"*string(i)); α = 1.0,color = coll[i])
    end

    mc.setprop!(vis["/Background"], "top_color", colorant"transparent")

    anim = mc.Animation(floor(Int,1/h))

    for k = 1:length(W)
        mc.atframe(anim, k) do
            for i = 1:N_bodies
                sym = Symbol("P"*string(i))
                mc.settransform!(vis[sym], mc.Translation(W[k][idx.z[i][1:3]]) ∘ mc.LinearMap(dc.dcm_from_q(W[k][idx.z[i][SA[4,5,6,7]]])))
            end
        end
    end
    mc.setanimation!(vis, anim)

end
