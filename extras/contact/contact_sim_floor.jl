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
import DifferentialProximity as dp
import Random
using Colors
using JLD2

include("variational_utils.jl")

function Ld(z₋,z₊)
    # write out all T - V given diff between z1 - z2
    r₋ = z₋[1:3]
    q₋ = z₋[4:7]

    r₊ = z₊[1:3]
    q₊ = z₊[4:7]

    v = (r₊ - r₋)/h
    ω = (2/h)*H'*L(q₋)'*q₊

    T = 0.5*m*dot(v,v) + 0.5*ω'*J*ω

    V = spring stuff

    return (h/2)*(T - V)
end

function DEL(z₋, z, z₊)

    # D2(Ld(z₋,z)) + D1(Ld(z,z₊))
    FD.jacobian(_z -> Ld(z₋, _z), z) + FD.jacobian(_z -> Ld(_z, z₊), z)

end


function trans_part(m,x1,x2,x3,Δt)
    (1/Δt)*m*(x2-x1) - (1/Δt)*m*(x3-x2) +  Δt*m*gravity
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
# const idx_z2 = 8:14
const idx_Δz1 = 1:6
# const idx_Δz2 = 7:12
const idx_s = 8
const idx_λ = 9
function DEL(z₋,z,z₊,J1,m1,h)
    single_DEL(z₋[idx_z1],z[idx_z1],z₊[idx_z1],J1,m1,h)
end
function update_pills!(z,P1)
    P1.r = SVector{3}(z[idx_z1[1:3]])
    P1.q = SVector{4}(z[idx_z1[4:7]])
end
function fd_α(P1,z)
    update_pills!(z,P1)
    α, x = dc.proximity_floor(P1; pdip_tol = 1e-6,basement = -10.0)
    return [(α - 1)]
end
function Gbar(z)
    Gbar1 = blockdiag(sparse(I(3)),sparse(G(z[idx_z1[4:7]])))
end
function contact_kkt(z₋,z,z₊,J1,m1,P1,h,κ)
    s = z₊[idx_s]
    λ = z₊[idx_λ]

    # jacobian of contact at middle step
    # update_pills!(z,P1)
    # _,_,D_state = dc.proximity_jacobian(P1,P2; pdip_tol = 1e-6)
    # D_α = reshape(D_state[4,:],1,14)
    D_α = FD2.finite_difference_jacobian(_z -> fd_α(P1,_z), z)
    D_α = D_α[:,1:7]
    @assert size(D_α) == (1,7)
    E = h*Diagonal([ones(3);0.5*ones(3)]) * (D_α*Gbar(z))'*[λ]

    # conatct stuff at + time step
    update_pills!(z₊,P1)
    α₊, _ = dc.proximity_floor(P1)

    [
    single_DEL(z₋[idx_z1],z[idx_z1],z₊[idx_z1],J1,m1,h) + E[1:6];#+ h*get_ft(z[idx_z1],p1,n1,λ);
    s - (α₊ - 1);#s - dp.proximity(dp_P1,dp_P2)#s - (α₊ - 1);
    s*λ - κ
    ]
end
# function contact_kkt_jacobian(z₋,z,z₊,J1,m1,P1, h)
#     s = z₊[idx_s]
#     λ = z₊[idx_λ]
#
#     # contact points at middle step to get contact jacobian
#     D_α = FD2.finite_difference_jacobian(_z -> fd_α(P1,_z), z)
#     E = h*Diagonal([ones(3);0.5*ones(3);zeros(2)])  * (D_α*Gbar(z))'*[λ]
#
#     # update pills with final step for the constraint term in the 3rd block
#     D_α₊ = FD2.finite_difference_jacobian(_z -> fd_α(P1,_z), z₊)
#
#     # three part jacobian
#     J1 = FD.jacobian(_z -> ([
#     single_DEL(z₋[idx_z1],z[idx_z1],_z[idx_z1],J1,m1,h);
#     single_DEL(z₋[idx_z2],z[idx_z2],_z[idx_z2],J2,m2,h)
#     ] + h*(D_α*Gbar(z))'*[_z[idx_λ]]), z₊)
#
#     J2 = [-D_α₊ 0 0]
#     J2[1,idx_s] = 1.0
#
#     J3 = zeros(1,9)
#     J3[1,idx_s] = λ
#     J3[1,idx_λ] = s
#
#     [J1;J2;J3]
# end
function linesearch(x,dx)
    α = min(0.99, minimum([dx[i]<0 ? -x[i]/dx[i] : Inf for i = 1:length(x)]))
end

function update_z(z,Δz)
    znew = deepcopy(z)
    znew[idx_z1[1:3]] += Δz[idx_Δz1[1:3]]
    znew[idx_z1[4:7]]  = L(znew[idx_z1[4:7]])*ρ(Δz[idx_Δz1[4:6]])
    znew[idx_s]       += Δz[idx_s - 1]
    znew[idx_λ]       += Δz[idx_λ - 1]
    return znew
end
function ncp_solve(z₋,z,J1,m1,h,P1)
    z₊ = copy(z) #+ .1*abs.(randn(length(z)))
    z₊[idx_s]+=1
    z₊[idx_λ]+=1
    @printf "iter    |∇ₓL|      |c|        κ          μ          α         αs        αλ\n"
    @printf "--------------------------------------------------------------------------\n"
    for i = 1:30
        rhs1 = -contact_kkt(z₋,z,z₊,J1,m1,P1,h,0)
        if norm(rhs1,Inf)<1e-10
            @info "success"
            return z₊
        end

        # finite diff for contact KKT
        D = FD2.finite_difference_jacobian(_z -> contact_kkt(z₋,z,_z,J1,m1,P1,h,0),z₊)
        # D = contact_kkt_jacobian(z₋,z,z₊,J1,m1,P1, h)

        # use G bar to handle quats and factorize
        F = factorize(D*blockdiag(Gbar(z₊),sparse(I(2))))

        # affine step
        Δa = F\rhs1
        αa = 0.99*min(linesearch(z₊[idx_s], Δa[idx_s .- 1]), linesearch(z₊[idx_λ], Δa[idx_λ .- 1]))

        # duality gap growth
        μ = dot(z₊[idx_s], z₊[idx_λ])
        μa = dot(z₊[idx_s] + αa*Δa[idx_s .- 1], z₊[idx_λ] + αa*Δa[idx_λ .- 1])
        σ = min(0.99,max(0,μa/μ))^3
        κ = max(min(σ*μ,1),1e-8)

        # centering step
        rhs2 = -contact_kkt(z₋,z,z₊,J1,m1,P1,h,κ)
        Δ = F\rhs2

        # update
        α1 = linesearch(z₊[idx_s], Δ[idx_s .- 1])
        α2 = linesearch(z₊[idx_λ], Δ[idx_λ .- 1])
        α = 0.99*min(α1, α2)

        # update
        z₊ = update_z(z₊,α*Δ)

        @printf("%3d    %9.2e  %9.2e  %9.2e  %9.2e  % 6.4f % 6.4f % 6.4f\n",
          i, norm(rhs1[1:6]), norm(rhs1[7:8]), κ, μ, α, α1, α2)

    end
    error("newton failed")
end


function viss()

    @load "/Users/kevintracy/.julia/dev/dc/extras/polytopes.jld2"

    # A1 = SMatrix{14,3}(A1)
    # b1 = SVector{14}(b1)
    # P1 = dc.Polytope(A1,b1)
    A2 = SMatrix{8,3}(A2)
    b2 = SVector{8}(b2)
    P1 = dc.Polytope(A2,b2)

    P1.r = SA[-20,0,4.0]

    m1 = 1.0
    J1 = Diagonal(SA[1,1,1.0])


    Random.seed!(1)
    h = 0.05
    v1 = 4*SA[1,0,0.0]
    ω1 = deg2rad(5)*randn(3)
    z0 = vcat(P1.r,P1.q)
    z1 = vcat(P1.r + h*v1,L(P1.q)*Expq(h*ω1))
    # z0 = vcat(P1.r,P1.q)
    # z1 = vcat(P1.r + h*v1, L(P1.q)*Expq(h*ω1))

    N = 200
    Z = [zeros(9) for i = 1:N]
    Z[1] = [z0;ones(2)]
    Z[2] = [z1;ones(2)]
    for i = 2:N-1
        # Z[i+1] = single_newton_solve(Z[i-1],Z[i],J1,m1,h)
        # Z[i+1] = newton_solve(Z[i-1],Z[i],J1,J2,m1,m2,h)
        Z[i+1] = ncp_solve(Z[i-1],Z[i],J1,m1,h, P1)

        if abs(norm(Z[i+1][4:7]) - 1) > 1e-10
            error("quat 1 is fucked")
        end
    end
    #
    # p1s = [SA[1,1,1.0] for i = 1:N]
    # p2s = [SA[1,1,1.0] for i = 1:N]
    # dp_p1s = [SA[1,1,1.0] for i = 1:N]
    # dp_p2s = [SA[1,1,1.0] for i = 1:N]
    # αs = zeros(N)
    # for i = 1:N
    #     update_pills!(Z[i],P1,P2)
    #     α, x = dc.proximity(P1, P2; pdip_tol = 1e-10)
    #     p1 = P1.r + (x - P1.r)/α
    #     p2 = P2.r + (x - P2.r)/α
    #     p1s[i] = p1
    #     p2s[i] = p2
    #     αs[i] = (α - 1)
    #
    #     # update_pills!(Z[i],dp_P1,dp_P2)
    #     # dp_p1s[i], dp_p2s[i], n1, n2 = dp.closest_points(dp_P1,dp_P2;pdip_tol = 1e-12)
    #
    # end
    #
    #
    # mat"
    # figure
    # hold on
    # plot($αs)
    # hold off
    # "
    #
    # sph_p1 = mc.HyperSphere(mc.Point(0,0,0.0), 0.1)
    # sph_p2 = mc.HyperSphere(mc.Point(0,0,0.0), 0.1)
    # mc.setobject!(vis[:p1], sph_p1,mc.MeshPhongMaterial(color = mc.RGBA(1.0,0,0,1.0)))
    # mc.setobject!(vis[:p2], sph_p2,mc.MeshPhongMaterial(color = mc.RGBA(1.0,0,0,1.0)))
    #
    # # dp_sph_p1 = mc.HyperSphere(mc.Point(0,0,0.0), 0.1)
    # # dp_sph_p2 = mc.HyperSphere(mc.Point(0,0,0.0), 0.1)
    # # mc.setobject!(vis[:dp_p1], dp_sph_p1,mc.MeshPhongMaterial(color = mc.RGBA(0.0,1.0,0,1.0)))
    # # mc.setobject!(vis[:dp_p2], dp_sph_p2,mc.MeshPhongMaterial(color = mc.RGBA(0.0,1.0,0,1.0)))
    #
    # # vis = mc.Visualizer()
    # # mc.open(vis)
    # c1 = [245, 155, 66]/255
    # c2 = [2,190,207]/255
    # dc.build_primitive!(vis, P1, :capsule; α = 1.0,color = mc.RGBA(c2..., 0.7))
    # dc.build_primitive!(vis, P2, :cone; α = 1.0,color = mc.RGBA(c1..., 0.7))
    #
    # mc.setprop!(vis["/Background"], "top_color", colorant"transparent")
    # dc.set_floor!(vis; x = 40, y = 40)
    #
    c1 = [245, 155, 66]/255
    dc.build_primitive!(vis, P1, :polytope1; α = 1.0,color = mc.RGBA(c1..., 1.0))
    dc.update_pose!(vis[:polytope1], P1)
    anim = mc.Animation(floor(Int,1/h))
    #
    for k = 1:length(Z)
        mc.atframe(anim, k) do
            r1 = Z[k][idx_z1[1:3]]
            q1 = Z[k][idx_z1[4:7]]
    #         r2 = Z[k][idx_z2[1:3]]
    #         q2 = Z[k][idx_z2[4:7]]
            mc.settransform!(vis[:polytope1],mc.Translation(r1) ∘ mc.LinearMap(mc.QuatRotation(q1)))
    #         mc.settransform!(vis[:cone],mc.Translation(r2) ∘ mc.LinearMap(mc.QuatRotation(q2)))
    #
    #         mc.settransform!(vis[:p1], mc.Translation(p1s[k]))
    #         mc.settransform!(vis[:p2], mc.Translation(p2s[k]))
    #
    #         # mc.settransform!(vis[:dp_p1], mc.Translation(dp_p1s[k]))
    #         # mc.settransform!(vis[:dp_p2], mc.Translation(dp_p2s[k]))
        end
    end
    mc.setanimation!(vis, anim)
    return nothing
end

#
vis = mc.Visualizer()
mc.open(vis)
viss()
