using Pkg
Pkg.activate(dirname(@__DIR__))
using LinearAlgebra
using StaticArrays
import ForwardDiff as FD
import FiniteDiff as FD2
using Printf
using SparseArrays
import MeshCat as mc
import DCD
using MATLAB
import DifferentialProximity as dp
import Random
# using MeshCat, GeometryBasics, CoordinateTransformations, Rotations
# using Colors
# using StaticArrays

include("variational_utils.jl")


function gen_inertia(R,L)
    ρ = 1.0
    r = R
    h = L
    v = h*π*r^2
    m = ρ*v
    Ixx = .5*m*r^2
    Iyy = (1/12)*m*(3*r^2 + h^2)
    Izz = Iyy
    m, Diagonal(SA[Ixx,Iyy,Izz])
end
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
const idx_z1 = 1:7
const idx_z2 = 8:14
const idx_Δz1 = 1:6
const idx_Δz2 = 7:12
const idx_s = 15
const idx_λ = 16
function DEL(z₋,z,z₊,J1,J2,m1,m2,h)
    [
    single_DEL(z₋[idx_z1],z[idx_z1],z₊[idx_z1],J1,m1,h);
    single_DEL(z₋[idx_z2],z[idx_z2],z₊[idx_z2],J2,m2,h)
    ]
end
function get_ft(z,p,n,λ)
    r = z[1:3]
    q = z[4:7]
    b_Q_n = DCD.dcm_from_q(SVector{4}(q))'
    f = λ*n
    τ = cross(b_Q_n*(p-r),b_Q_n*f)
    return [f;τ]
end
function get_states(z)
    r1 = z[idx_z1[1:3]]
    q1 = z[idx_z1[4:7]]
    r2 = z[idx_z2[1:3]]
    q2 = z[idx_z2[4:7]]
    r1,q1,r2,q2
end
function update_pills!(z,P1,P2)
    P1.r = SVector{3}(z[idx_z1[1:3]])
    P1.q = SVector{4}(z[idx_z1[4:7]])
    P2.r = SVector{3}(z[idx_z2[1:3]])
    P2.q = SVector{4}(z[idx_z2[4:7]])
end
function fd_α(P1,P2,z)
    update_pills!(z,P1,P2)
    α, x = DCD.proximity(P1,P2; pdip_tol = 1e-8)
    return (α - 1)
end
function Gbar(z)
    Gbar1 = blockdiag(sparse(I(3)),sparse(G(z[idx_z1[4:7]])))
    Gbar2 = blockdiag(sparse(I(3)),sparse(G(z[idx_z2[4:7]])))
    Gbar = blockdiag(Gbar1,Gbar2)
end
function contact_kkt(z₋,z,z₊,J1,J2,m1,m2,P1,P2,dp_P1, dp_P2, h,κ)
    s = z₊[idx_s]
    λ = z₊[idx_λ]

    # jacobian of contact at middle step
    update_pills!(z,P1,P2)
    _,_,D_state = DCD.proximity_jacobian(P1,P2; pdip_tol = 1e-6)
    D_α = reshape(D_state[4,:],1,14)
    E = h*(D_α*Gbar(z))'*[λ]

    # conatct stuff at + time step
    update_pills!(z₊,P1,P2)
    α₊, _ = DCD.proximity(P1,P2)

    [
    single_DEL(z₋[idx_z1],z[idx_z1],z₊[idx_z1],J1,m1,h) + E[1:6];#+ h*get_ft(z[idx_z1],p1,n1,λ);
    single_DEL(z₋[idx_z2],z[idx_z2],z₊[idx_z2],J2,m2,h) + E[7:12];#h*get_ft(z[idx_z2],p2,n2,λ);
    s - (α₊ - 1);#s - dp.proximity(dp_P1,dp_P2)#s - (α₊ - 1);
    s*λ - κ
    ]
end
function contact_kkt_jacobian(z₋,z,z₊,J1,J2,m1,m2,P1,P2,dp_P1, dp_P2, h,κ)
    s = z₊[idx_s]
    λ = z₊[idx_λ]

    # contact points at middle step to get contact jacobian
    update_pills!(z,P1,P2)
    _,_,D_state = DCD.proximity_jacobian(P1,P2; pdip_tol = 1e-6)
    D_α = reshape(D_state[4,:],1,14)
    E = h*(D_α*Gbar(z))'*[λ]

    # update pills with final step for the constraint term in the 3rd block
    update_pills!(z₊,P1,P2)
    α₊, _, D_state₊ = DCD.proximity_jacobian(P1,P2)
    D_α₊ = reshape(D_state₊[4,:],1,14)

    # three part jacobian
    J1 = FD.jacobian(_z -> ([
    single_DEL(z₋[idx_z1],z[idx_z1],_z[idx_z1],J1,m1,h);
    single_DEL(z₋[idx_z2],z[idx_z2],_z[idx_z2],J2,m2,h)
    ] + h*(D_α*Gbar(z))'*[_z[idx_λ]]), z₊)

    J2 = [-D_α₊ 0 0]
    J2[1,idx_s] = 1.0

    J3 = zeros(1,16)
    J3[1,idx_s] = λ
    J3[1,idx_λ] = s

    [J1;J2;J3]
end
function linesearch(x,dx)
    α = min(0.99, minimum([dx[i]<0 ? -x[i]/dx[i] : Inf for i = 1:length(x)]))
end

function update_z(z,Δz)
    znew = deepcopy(z)
    znew[idx_z1[1:3]] += Δz[idx_Δz1[1:3]]
    znew[idx_z1[4:7]]  = L(znew[idx_z1[4:7]])*ρ(Δz[idx_Δz1[4:6]])
    znew[idx_z2[1:3]] += Δz[idx_Δz2[1:3]]
    znew[idx_z2[4:7]]  = L(znew[idx_z2[4:7]])*ρ(Δz[idx_Δz2[4:6]])
    znew[idx_s]       += Δz[idx_s - 2]
    znew[idx_λ]       += Δz[idx_λ - 2]
    return znew
end
function ncp_solve(z₋,z,J1,J2,m1,m2,h,P1,P2,dp_P1, dp_P2)
    z₊ = copy(z) #+ .1*abs.(randn(length(z)))
    z₊[idx_s]+=1
    z₊[idx_λ]+=1
    @printf "iter    |∇ₓL|      |c|        κ          μ          α         αs        αλ\n"
    @printf "--------------------------------------------------------------------------\n"
    for i = 1:30
        rhs1 = -contact_kkt(z₋,z,z₊,J1,J2,m1,m2,P1,P2,dp_P1, dp_P2, h,0)
        if norm(rhs1)<1e-6
            @info "success"
            return z₊
        end

        # finite diff for contact KKT
        # D = FD2.finite_difference_jacobian(_z -> contact_kkt(z₋,z,_z,J1,J2,m1,m2,P1,P2,dp_P1, dp_P2, h,0),z₊)
        D = contact_kkt_jacobian(z₋,z,z₊,J1,J2,m1,m2,P1,P2,dp_P1, dp_P2, h,0)

        # use G bar to handle quats and factorize
        F = factorize(D*blockdiag(Gbar(z₊),sparse(I(2))))

        # affine step
        Δa = F\rhs1
        αa = 0.99*min(linesearch(z₊[idx_s], Δa[idx_s .- 2]), linesearch(z₊[idx_λ], Δa[idx_λ .- 2]))

        # duality gap growth
        μ = dot(z₊[idx_s], z₊[idx_λ])
        μa = dot(z₊[idx_s] + αa*Δa[idx_s .- 2], z₊[idx_λ] + αa*Δa[idx_λ .- 2])
        σ = min(0.99,max(0,μa/μ))^3
        κ = max(min(σ*μ,1),1e-8)

        # centering step
        rhs2 = -contact_kkt(z₋,z,z₊,J1,J2,m1,m2,P1,P2,dp_P1, dp_P2,h,κ)
        Δ = F\rhs2

        # update
        α1 = linesearch(z₊[idx_s], Δ[idx_s .- 2])
        α2 = linesearch(z₊[idx_λ], Δ[idx_λ .- 2])
        α = 0.99*min(α1, α2)

        # update
        z₊ = update_z(z₊,α*Δ)

        @printf("%3d    %9.2e  %9.2e  %9.2e  %9.2e  % 6.4f % 6.4f % 6.4f\n",
          i, norm(rhs1[1:12]), norm(rhs1[13:14]), κ, μ, α, α1, α2)

    end
    error("newton failed")
end


# vis = Visualizer()
function viss()

    dp_P1 = dp.create_capsule(:quat)
    dp_P2 = dp.create_capsule(:quat)
    P1 = DCD.Capsule(1.3,2.0)
    # P2 = DCD.Capsule(1.0,4.0)
    P2 = DCD.Cone(3.0,deg2rad(22))

    dp_P1.R = 1.3
    dp_P2.R = 1.0
    dp_P1.L = 2.0
    dp_P2.L = 4
    # P1.r = SA[1,1,2.0]
    # P2.r = SA[8.0,0,0]
    # P1.r = SA[-10,-10,-5]
    # P2.r = SA[10,10,5]
    # P1.q = normalize(@SVector randn(4))
    # P2.q = normalize(@SVector randn(4))
    P1.r = SA[-4,0,0.0]
    P2.r = SA[4,0,0.0]
    # P1.q = normalize(@SVector randn(4))
    # P2.q = normalize(@SVector randn(4))

    m1, J1 = gen_inertia(1.3,2.0)
    m2, J2 = gen_inertia(1.3,2.0)


    Random.seed!(1)
    h = 0.01
    v1 = 4*SA[1,0,0.0]
    v2 = 4*SA[-1,0,0.0]
    ω1 = deg2rad(40)*randn(3)
    ω2 = deg2rad(40)*randn(3)
    z0 = vcat(P1.r,P1.q,P2.r,P2.q)
    z1 = vcat(P1.r + h*v1,L(P1.q)*Expq(h*ω1),P2.r + v2*h,L(P2.q)*Expq(h*ω2))
    # z0 = vcat(P1.r,P1.q)
    # z1 = vcat(P1.r + h*v1, L(P1.q)*Expq(h*ω1))

    N = 300
    Z= [zeros(16) for i = 1:N]
    Z[1] = [z0;ones(2)]
    Z[2] = [z1;ones(2)]
    for i = 2:N-1
        # Z[i+1] = single_newton_solve(Z[i-1],Z[i],J1,m1,h)
        # Z[i+1] = newton_solve(Z[i-1],Z[i],J1,J2,m1,m2,h)
        Z[i+1] = ncp_solve(Z[i-1],Z[i],J1,J2,m1,m2,h, P1, P2, dp_P1, dp_P2)

        if abs(norm(Z[i+1][4:7]) - 1) > 1e-13
            error("quat 1 is fucked")
        end
        if abs(norm(Z[i+1][11:14]) - 1) > 1e-13
            error("quat 2 is fucked")
        end
    end

    p1s = [SA[1,1,1.0] for i = 1:N]
    p2s = [SA[1,1,1.0] for i = 1:N]
    dp_p1s = [SA[1,1,1.0] for i = 1:N]
    dp_p2s = [SA[1,1,1.0] for i = 1:N]
    αs = zeros(N)
    for i = 1:N
        update_pills!(Z[i],P1,P2)
        α, x = DCD.proximity(P1, P2; pdip_tol = 1e-10)
        p1 = P1.r + (x - P1.r)/α
        p2 = P2.r + (x - P2.r)/α
        p1s[i] = p1
        p2s[i] = p2
        αs[i] = (α - 1)

        # update_pills!(Z[i],dp_P1,dp_P2)
        # dp_p1s[i], dp_p2s[i], n1, n2 = dp.closest_points(dp_P1,dp_P2;pdip_tol = 1e-12)

    end


    mat"
    figure
    hold on
    plot($αs)
    hold off
    "

    sph_p1 = mc.HyperSphere(mc.Point(0,0,0.0), 0.1)
    sph_p2 = mc.HyperSphere(mc.Point(0,0,0.0), 0.1)
    mc.setobject!(vis[:p1], sph_p1,mc.MeshPhongMaterial(color = mc.RGBA(1.0,0,0,1.0)))
    mc.setobject!(vis[:p2], sph_p2,mc.MeshPhongMaterial(color = mc.RGBA(1.0,0,0,1.0)))

    # dp_sph_p1 = mc.HyperSphere(mc.Point(0,0,0.0), 0.1)
    # dp_sph_p2 = mc.HyperSphere(mc.Point(0,0,0.0), 0.1)
    # mc.setobject!(vis[:dp_p1], dp_sph_p1,mc.MeshPhongMaterial(color = mc.RGBA(0.0,1.0,0,1.0)))
    # mc.setobject!(vis[:dp_p2], dp_sph_p2,mc.MeshPhongMaterial(color = mc.RGBA(0.0,1.0,0,1.0)))

    # vis = mc.Visualizer()
    # mc.open(vis)
    c1 = [245, 155, 66]/255
    c2 = [2,190,207]/255
    DCD.build_primitive!(vis, P1, :capsule; α = 1.0,color = mc.RGBA(c2..., 0.7))
    DCD.build_primitive!(vis, P2, :cone; α = 1.0,color = mc.RGBA(c1..., 0.7))

    anim = mc.Animation(floor(Int,1/h))

    for k = 1:length(Z)
        mc.atframe(anim, k) do
            r1 = Z[k][idx_z1[1:3]]
            q1 = Z[k][idx_z1[4:7]]
            r2 = Z[k][idx_z2[1:3]]
            q2 = Z[k][idx_z2[4:7]]
            mc.settransform!(vis[:capsule],mc.Translation(r1) ∘ mc.LinearMap(mc.QuatRotation(q1)))
            mc.settransform!(vis[:cone],mc.Translation(r2) ∘ mc.LinearMap(mc.QuatRotation(q2)))

            mc.settransform!(vis[:p1], mc.Translation(p1s[k]))
            mc.settransform!(vis[:p2], mc.Translation(p2s[k]))

            # mc.settransform!(vis[:dp_p1], mc.Translation(dp_p1s[k]))
            # mc.settransform!(vis[:dp_p2], mc.Translation(dp_p2s[k]))
        end
    end
    mc.setanimation!(vis, anim)
    # # vis = Visualizer()
    # anim = MeshCat.Animation(floor(Int,1/h))
    # build_pill!(vis, :pill_1, P1.L, P1.R)
    # build_pill!(vis, :pill_2, P2.L, P2.R)
    # build_particles!(vis)
    #
    # for i = 1:length(Z)
    #     atframe(anim, i) do
    #         r1 = Z[i][idx_z1[1:3]]
    #         q1 = Z[i][idx_z1[4:7]]
    #         r2 = Z[i][idx_z2[1:3]]
    #         q2 = Z[i][idx_z2[4:7]]
    #         settransform!(vis[:pill_1], compose(Translation(r1), LinearMap(QuatRotation(q1))))
    #         settransform!(vis[:pill_2], compose(Translation(r2), LinearMap(QuatRotation(q2))))
    #
    #         P1.r = r1; P1.q = q1; P2.r = r2; P2.q = q2;
    #         p1,p2,n1,n2 = dp.closest_points(P1,P2;pdip_tol = 1e-12)
    #         settransform!(vis[:p1], Translation(p1))
    #         settransform!(vis[:p2], Translation(p2))
    #     end
    # end
    # setanimation!(vis, anim)

    return nothing
end


vis = mc.Visualizer()
mc.open(vis)
viss()
