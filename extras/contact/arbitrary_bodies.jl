using Pkg
Pkg.activate(dirname(@__DIR__))
using LinearAlgebra
using StaticArrays
import ForwardDiff as FD
import FiniteDiff as FD2
using Printf
using SparseArrays
import MeshCat as mc
import DCOL as dc
using MATLAB
import Random
using Colors
using Printf
using Combinatorics


include("/Users/kevintracy/.julia/dev/DCOL/extras/contact/variational_utils.jl")
function trans_part(m,x1,x2,x3,Δt)
    (1/Δt)*m*(x2-x1) - (1/Δt)*m*(x3-x2) #+  Δt*m*gravity
end

function rot_part(J,q1,q2,q3,Δt)
    (2.0/Δt)*G(q2)'*L(q1)*H*J*H'*L(q1)'*q2 + (2.0/Δt)*G(q2)'*T*R(q3)'*H*J*H'*L(q2)'*q3
end

function single_DEL(z₋,z,z₊,J,m,h)

    p₋ = z₋[1:3]
    q₋ = z₋[4:7]
    p = z[1:3]
    q = z[4:7]
    p₊ = z₊[1:3]
    q₊ = z₊[4:7]
    [trans_part(m,p₋,p,p₊,h);rot_part(J,q₋,q,q₊,h)]
end
function create_indexing(N_bodies)
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
    for i = 1:idx.N_bodies
        P[i].r = w[idx.z[i][SA[1,2,3]]]
        P[i].q = normalize(w[idx.z[i][SA[4,5,6,7]]])
    end
    nothing
end

function contacts(P,idx)
    # returns α - 1 for all contacts
    [(dc.proximity(P[i],P[j])[1] - 1) for (i,j) in idx.interactions]
    # as = zeros(idx.N_interactions)
    # for k = 1:idx.N_interactions
    #     i,j = idx.interactions[k]
    #     @show P[i]
    #     @show P[j]
    #     as[i] = dc.proximity(P[i],P[j])[1] - 1
    # end
    # as
end
function linesearch(x,dx)
    α = min(0.99, minimum([dx[i]<0 ? -x[i]/dx[i] : Inf for i = 1:length(x)]))
end
function Gbar_2_body(z,idx_z1, idx_z2)
    Gbar1 = blockdiag(sparse(I(3)),sparse(G(z[idx_z1[4:7]])))
    Gbar2 = blockdiag(sparse(I(3)),sparse(G(z[idx_z2[4:7]])))
    Gbar = Matrix(blockdiag(Gbar1,Gbar2))
end
function Gbar(w,idx)
    G1 = (blockdiag([blockdiag(sparse(I(3)),sparse(G(w[idx.z[i][4:7]]))) for i = 1:idx.N_bodies]...))
    Matrix(blockdiag(G1,sparse(I(2*length(idx.interactions)))))
end

function update_se3(state, delta)
    [
    state[SA[1,2,3]] + delta[SA[1,2,3]];
    L(state[SA[4,5,6,7]]) * ρ(delta[SA[4,5,6]])
    ]
end
function update_w(w,Δ,idx)
    wnew = deepcopy(w)
    for i = 1:idx.N_bodies
        wnew[idx.z[i]] = update_se3(w[idx.z[i]], Δ[idx.Δz[i]])
    end
    wnew[idx.s] += Δ[idx.Δs]
    wnew[idx.λ] += Δ[idx.Δλ]
    wnew
end
const τ_mod = Diagonal(kron(ones(2),[ones(3);0.5*ones(3)]))

function contact_kkt(w₋, w, w₊, P, inertias, masses, idx, κ)
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

    # now get contact stuff for + time step
    update_objects!(P,w₊,idx)
    αs = contacts(P,idx)

    [
    vcat(DELs...);
    s - αs;
    (s .* λ) .- κ;
    ]
end
function contact_kkt_no_α(w₋, w, w₊, P, inertias, masses, idx, κ)
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

    # now get contact stuff for + time step
    # update_objects!(P,w₊,idx)
    # αs = contacts(P,idx)

    [
    vcat(DELs...);
    s; #- αs;
    (s .* λ) .- κ;
    ]
end
function ncp_solve(w₋, w, P, inertias, masses, idx)
    w₊ = copy(w) #+ .1*abs.(randn(length(z)))
    w₊[idx.s] .+= .1
    w₊[idx.λ] .+= .1

    @printf "iter    |∇ₓL|      |c|        κ          μ          α         αs        αλ\n"
    @printf "--------------------------------------------------------------------------\n"

    # @info "inside NCP solve"
    for i = 1:30
        rhs1 = -contact_kkt(w₋, w, w₊, P, inertias, masses, idx, 0)
        if norm(rhs1)<1e-6
            @info "success"
            return w₊
        end

        # finite diff for contact KKT
        # D = FD2.finite_difference_jacobian(_w -> contact_kkt(w₋, w, _w, P, inertias, masses, idx, 0.0),w₊)
        D = FD.jacobian(_w -> contact_kkt_no_α(w₋, w, _w, P, inertias, masses, idx, 0.0),w₊)

        # add jacobians for all our stuff
        update_objects!(P,w₊,idx)
        for k = 1:idx.N_interactions
            i,j = idx.interactions[k]
            (dc.proximity(P[i],P[j])[1] - 1)
            _,_,D_α = dc.proximity_jacobian(P[i],P[j])
            D_α = reshape(D_α[4,:],1,14)
            D[idx.α[k],idx.z[i]] = -D_α[1,idx.z[1]]
            D[idx.α[k],idx.z[j]] = -D_α[1,idx.z[2]]
        end



        # use G bar to handle quats and factorize
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

        # update
        α1 = linesearch(w₊[idx.s], Δ[idx.Δs])
        α2 = linesearch(w₊[idx.λ], Δ[idx.Δλ])
        α = 0.99*min(α1, α2)

        # update
        w₊ = update_w(w₊,α*Δ,idx)

        @printf("%3d    %9.2e  %9.2e  %9.2e  %9.2e  % 6.4f % 6.4f % 6.4f\n",
          i, norm(rhs1[1:12]), norm(rhs1[13:14]), κ, μ, α, α1, α2)

    end
    error("newton failed")
end

N_bodies = 9 # TODO: don't forget to change this back

idx = create_indexing(N_bodies)
h = 0.05
P = [dc.Cylinder(0.2,6.0), dc.Capsule(0.4,3.0), dc.Sphere(1.5),
dc.Cylinder(0.2,6.0), dc.Capsule(0.4,3.0), dc.Sphere(1.5),
dc.Cylinder(0.2,6.0), dc.Capsule(0.4,3.0), dc.Sphere(1.5) ]
@assert length(P) == N_bodies

masses = ones(N_bodies)
inertias = [Diagonal(SA[1,2,3.0]) for i = 1:N_bodies]

# rs = [SA[-2,1,2.0], SA[1,-1.0,2.5], SA[0,0,6.0]]
rs = [10*(@SVector randn(3)) for i = 1:N_bodies]
qs = [SA[1,0,0,0.0] for i = 1:N_bodies]

w0 = vcat([[rs[i];qs[i]] for i = 1:N_bodies]..., zeros(2*idx.N_interactions))

vs =  [SA[1,1,1.0] for i = 1:N_bodies]
for i = 1:N_bodies
    vs[i] = -.5*rs[i]
end

ωs = [deg2rad.(10*(@SVector randn(3))) for i = 1:N_bodies]

r2s = [(rs[i] + h*vs[i]) for i = 1:N_bodies]
q2s = [(L(qs[i])*Expq(h*ωs[i])) for i = 1:N_bodies]
w1 = vcat([[r2s[i];q2s[i]] for i = 1:N_bodies]..., zeros(2*idx.N_interactions))

N = 100
W = [zeros(length(w0)) for i = 1:N]
W[1] = w0
W[2] = w1

for i = 2:N-1
    W[i+1] = ncp_solve(W[i-1], W[i], P, inertias, masses, idx)
end

vis = mc.Visualizer()
mc.open(vis)


for i = 1:N_bodies
    dc.build_primitive!(vis, P[i], Symbol("P"*string(i)); α = 1.0,color = mc.RGBA(normalize(abs.(randn(3)))..., 1.0))
end

# mc.setprop!(vis["/Background"], "top_color", colorant"transparent")
# dc.set_floor!(vis; x = 40, y = 40)

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
