using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using DCOL
Pkg.activate(@__DIR__)
Pkg.instantiate()

using LinearAlgebra
using Printf
using StaticArrays
import ForwardDiff as FD
import MeshCat as mc
import DCOL as dc
import Random
using Colors

# add altro
include(joinpath(@__DIR__,"simple_altro.jl"))

function skew(ω::Vector{T}) where {T}
    return [0 -ω[3] ω[2];
            ω[3] 0 -ω[1];
            -ω[2] ω[1] 0]
end
function rigid_body_dynamics(J,m,x,u)
    r = x[1:3]
    v = x[4:6]
    p = x[7:9]
    ω = x[10:12]

    f = u[1:3]
    τ = u[4:6]

    [
        v
        f/m
        ((1+norm(p)^2)/4) *(   I + 2*(skew(p)^2 + skew(p))/(1+norm(p)^2)   )*ω
        J\(τ - cross(ω,J*ω))
    ]
end
function dynamics(p::NamedTuple,x,u,k)
    rigid_body_dynamics(p.J,p.m,x,u)
end
function discrete_dynamics(p::NamedTuple,x,u,k)
    k1 = p.dt*dynamics(p,x,        u, k)
    k2 = p.dt*dynamics(p,x + k1/2, u, k)
    k3 = p.dt*dynamics(p,x + k2/2, u, k)
    k4 = p.dt*dynamics(p,x + k3, u, k)
    x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
end
function ineq_con_u(p,u)
    [u-p.u_max;-u + p.u_min]
end
function ineq_con_u_jac(params,u)
    nu = params.nu
    Array(float([I(nu);-I(nu)]))
end
function ineq_con_x(p,x)
    p.P_vic.r = SVector{3}(x[1:3])
    p.P_vic.p = SVector{3}(x[7:9])
    contacts= [(1 - dc.proximity(p.P_vic, p.P_obs[i])[1]) for i = 1:length(p.P_obs)]
    vcat(contacts...)
end
function ineq_con_x_jac(p,x)
    p.P_vic.r = SVector{3}(x[1:3])
    p.P_vic.p = SVector{3}(x[7:9])
    contact_J = [ [-reshape(dc.proximity_jacobian(p.P_vic, p.P_obs[i])[3][4,1:3],1,3) zeros(1,3) -reshape(dc.proximity_jacobian(p.P_vic, p.P_obs[i])[3][4,4:6],1,3) zeros(1,3)] for i = 1:length(p.P_obs)]
    vcat(contact_J...)
end

function linear_interp(dt,x0,xg,N)
    Δp = (xg[1:3] - x0[1:3])
    positions = [((i - 1)*(Δp/(N-1)) + x0[1:3]) for i = 1:N]
    Δp = (xg[7:9] - x0[7:9])
    attitudes = [((i - 1)*(Δp/(N-1)) + x0[7:9]) for i = 1:N]
    @assert positions[1] ≈ x0[1:3]
    @assert positions[N] ≈ xg[1:3]
    @assert attitudes[1] ≈ x0[7:9]
    @assert attitudes[N] ≈ xg[7:9]
    velocities = [Δp/(N*dt) for i = 1:N]
    [[positions[i];velocities[i];attitudes[i]; zeros(3)] for i = 1:N]
end

vis = mc.Visualizer()
mc.open(vis)
let
    nx = 12
    nu = 6
    N = 60
    dt = 0.1
    x0 = [-4,-7,9,0.0,0.0,0.0,0,0,0,0,0,0]
    xg = [-4.5,7,3,0,0,0.0,.0,.0,.0,0,0,0]
    Xref = linear_interp(dt,x0,xg,N)
    Uref = [zeros(nu) for i = 1:N]
    Q = Diagonal(ones(nx))
    Qf = Diagonal(ones(nx))
    R = Diagonal(ones(nu))
    R = Diagonal([ones(3);100*ones(3)])

    P_vic = dc.ConeMRP(2.0, deg2rad(22))
    mass,inertia = dc.mass_properties(P_vic)

    P_obs = [dc.create_rect_prism(10.0, 10.0, 1.0)[1],
             dc.create_rect_prism(10.0, 10.0, 1.0)[1],
             dc.create_rect_prism(4.1, 4.1, 1.1)[1],
             dc.create_rect_prism(4.1, 4.1, 1.1)[1]]

    P_obs[1].r = SA[-6,0,5.0]
    P_obs[1].p = dc.mrp_from_q(SA[cos(pi/4),sin(pi/4),0,0])
    P_obs[2].r = SA[6,0,5.0]
    P_obs[2].p = dc.mrp_from_q(SA[cos(pi/4),sin(pi/4),0,0])
    P_obs[3].r = SA[0,0,2.05]
    P_obs[3].p = dc.mrp_from_q(SA[cos(pi/4),sin(pi/4),0,0])
    P_obs[4].r = SA[0,0,7.96]
    P_obs[4].p = dc.mrp_from_q(SA[cos(pi/4),sin(pi/4),0,0])

    # control limits
    u_min = -20*ones(nu)
    u_max =  20*ones(nu)

    # state limits
    x_min = -20*ones(nx)
    x_max =  20*ones(nx)


    # number of constraints in ineq_con_x
    ncx = length(P_obs)
    ncu = 2*nu

    params = (
        nx = nx,
        nu = nu,
        ncx = ncx,
        ncu = ncu,
        N = N,
        Q = Q,
        R = R,
        Qf = Qf,
        u_min = u_min,
        u_max = u_max,
        x_min = x_min,
        x_max = x_max,
        Xref = Xref,
        Uref = Uref,
        dt = dt,
        m = mass,
        J = inertia,
        P_obs = P_obs,
        P_vic = P_vic
    );


    X = [deepcopy(x0) for i = 1:N]
    using Random
    Random.seed!(2)
    U = [.01*randn(nu) for i = 1:N-1]

    Xn = deepcopy(X)
    Un = deepcopy(U)


    P = [zeros(nx,nx) for i = 1:N]   # cost to go quadratic term
    p = [zeros(nx) for i = 1:N]      # cost to go linear term
    d = [zeros(nu) for i = 1:N-1]    # feedforward control
    K = [zeros(nu,nx) for i = 1:N-1] # feedback gain
    Xhist = iLQR(params,X,U,P,p,K,d,Xn,Un;atol=1e-1,max_iters = 3000,verbose = true,ρ = 1e0, ϕ = 10.0 )

    animation = true

    if animation
        mc.setprop!(vis["/Background"], "top_color", colorant"transparent")
        mc.setprop!(vis["/Lights/AmbientLight/<object>"], "intensity", 0.9)
        mc.setprop!(vis["/Lights/PointLightPositiveX/<object>"], "intensity", 0.0)
        mc.setprop!(vis["/Lights/FillLight/<object>"], "intensity", 0.25)
        mc.setvisible!(vis["/Grid"],false)
        mc.setvisible!(vis["/Axes"],false)
        dc.set_floor!(vis; darkmode = false, x = 20, y = 20)

        coll = shuffle(range(HSVA(0,0.7,.75,0.3), stop=HSVA(-200,0.7,.75,0.3), length=4))
        for i = 1:length(P_obs)
            dc.build_primitive!(vis, P_obs[i], Symbol("P"*string(i)); α = 1.0,color = coll[i])
            dc.update_pose!(vis[Symbol("P"*string(i))],P_obs[i])
        end

        # build actual vehicle
        c2 = [245, 155, 66]/255
        dc.build_primitive!(vis, P_vic, :vic; α = 1.0,color = mc.RGBA(c2..., 1.0))

        anim = mc.Animation(floor(Int,1/dt))
        for k = 1:N
            mc.atframe(anim, k) do
                P_vic.r = X[k][SA[1,2,3]]
                P_vic.p = X[k][SA[7,8,9]]
                dc.update_pose!(vis[:vic], P_vic)
            end
        end
        mc.setanimation!(vis, anim)
    else


        mc.setprop!(vis["/Lights/AmbientLight/<object>"], "intensity", 0.9)
        mc.setprop!(vis["/Lights/PointLightPositiveX/<object>"], "intensity", 0.0)
        mc.setprop!(vis["/Lights/FillLight/<object>"], "intensity", 0.25)
        mc.setvisible!(vis["/Grid"],true)
        mc.setvisible!(vis["/Background"],false)
        mc.setvisible!(vis["/Axes"],false)


        coll = shuffle(range(HSVA(0,0.7,.75,0.3), stop=HSVA(-200,0.7,.75,0.3), length=4))
        for i = 1:length(P_obs)
            dc.build_primitive!(vis, P_obs[i], Symbol("P"*string(i)); α = 1.0,color = coll[i])
            dc.update_pose!(vis[Symbol("P"*string(i))],P_obs[i])
        end


        # build actual vehicle
        # dc.build_primitive!(vis, P_vic, :vic; α = 1.0,color = mc.RGBA(normalize(abs.(randn(3)))..., 1.0))

        # # visualize trajectory
        traj_alphas = range(.1,1,length = length(Xhist))
        for i = length(Xhist)
            dc.vis_traj!(vis, "traj"*string(i), Xhist[i]; R = 0.07, color = mc.RGBA(1.0, 0.0, 0.0, traj_alphas[i]))
        end

        c2 = [245, 155, 66]/255
        for k = [1,15,30,45,60]
            dc.build_primitive!(vis, P_vic, "vic"*string(k); α = 1.0,color = mc.RGBA(c2..., 0.7))
            mc.settransform!(vis["vic"*string(k)], mc.Translation(X[k][1:3]) ∘ mc.LinearMap(dc.dcm_from_mrp(X[k][SA[7,8,9]])))
        end
        # anim = mc.Animation(floor(Int,1/dt))
        # for k = 1:N
        #     mc.atframe(anim, k) do
        #         mc.settransform!(vis[:vic], mc.Translation(X[k][1:3]) ∘ mc.LinearMap(dc.dcm_from_mrp(X[k][SA[7,8,9]])))
        #         # dc.update_pose!(vis[:vic], P_vic)
        #     end
        # end
        # mc.setanimation!(vis, anim)
    end
end
