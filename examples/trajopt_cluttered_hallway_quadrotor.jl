using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using DCOL
Pkg.activate(@__DIR__)
Pkg.instantiate()

using LinearAlgebra
using Printf
using StaticArrays
using SparseArrays
import ForwardDiff as FD
import MeshCat as mc
import DCOL as dc
import Random
using Colors
import Random
using JLD2
include(joinpath(@__DIR__,"simple_altro.jl"))


function skew(ω::Vector{T}) where {T}
    return [0 -ω[3] ω[2];
            ω[3] 0 -ω[1];
            -ω[2] ω[1] 0]
end
function dynamics(params::NamedTuple,x,u,k_iter)
    # quadrotor dynamics with an MRP for attitude
    r = x[1:3]
    v = x[4:6]
    p = x[7:9]
    ω = x[10:12]

    Q = dc.dcm_from_mrp(SVector{3}(p))

    mass=0.5
    J=Diagonal([0.0023, 0.0023, 0.004])
    gravity=[0,0,-9.81]
    L=0.1750
    kf=1.0
    km=0.0245

    w1 = u[1]
    w2 = u[2]
    w3 = u[3]
    w4 = u[4]

    F1 = max(0,kf*w1)
    F2 = max(0,kf*w2)
    F3 = max(0,kf*w3)
    F4 = max(0,kf*w4)
    F = [0., 0., F1+F2+F3+F4] #total rotor force in body frame

    M1 = km*w1;
    M2 = km*w2;
    M3 = km*w3;
    M4 = km*w4;
    τ = [L*(F2-F4), L*(F3-F1), (M1-M2+M3-M4)] #total rotor torque in body frame

    f = mass*gravity + Q*F # forces in world frame

    [
        v
        f/mass
        ((1+norm(p)^2)/4) *(   I + 2*(skew(p)^2 + skew(p))/(1+norm(p)^2)   )*ω
        J\(τ - cross(ω,J*ω))
    ]
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
    # update P_vic struct with the position and attitude
    p.P_vic.r = SVector{3}(x[1:3])
    p.P_vic.p = SVector{3}(x[7:9])

    # (1 - α ≤ 0) for all pairs of P_vic and P_obs[i]
    return [(1 - dc.proximity(p.P_vic, p.P_obs[i])[1]) for i = 1:length(p.P_obs)]
end
function ineq_con_x_jac(p,x)
    # update P_vic struct with the position and attitude
    p.P_vic.r = SVector{3}(x[1:3])
    p.P_vic.p = SVector{3}(x[7:9])

    # calculate all of the jacobians from DCOL
    Js = [dc.proximity_jacobian(p.P_vic, p.P_obs[i])[3] for i = 1:length(p.P_obs)]

    # pull out the stuff we need for each constraint and stack it up
    contact_J = [ [-reshape(Js[i][4,1:3],1,3) zeros(1,3) -reshape(Js[i][4,4:6],1,3) zeros(1,3)] for i = 1:length(p.P_obs)]
    return vcat(contact_J...)
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
    nu = 4
    N = 100
    dt = 0.08
    x0 = [-8;0;4;0;0;0.0;zeros(6)]
    xg = [8;0;4;0;0;0.0;zeros(6)]
    Xref = linear_interp(dt,x0,xg,N)
    Uref = [(9.81*0.5/4)*ones(nu) for i = 1:N]
    Q = Diagonal(ones(nx))
    Qf = Diagonal(ones(nx))
    R = 1*Diagonal(ones(nu))
    obstacle = zeros(3)
    obstacle_R = 2.0

    # create out "quadrotor" where we model it as a sphere
    P_vic = dc.SphereMRP(0.25)

    # @load "/Users/kevintracy/.julia/dev/DifferentialProximity/extras/polyhedra_plotting/polytopes.jld2"
    path_str = joinpath(dirname(@__DIR__),"test/example_socps/polytopes.jld2")
    f = jldopen(path_str)
    A1 = SMatrix{14,3}(f["A1"])
    b1 = SVector{14}(f["b1"])
    A2 = SMatrix{8,3}(f["A2"])
    b2 = SVector{8}(f["b2"])

    P_bot = dc.create_rect_prism(20, 5.0, 0.2)[1]
    P_bot.r = [0,0,0.9]
    P_top = dc.create_rect_prism(20, 5.0, 0.2)[1]
    P_top.r = [0,0,6.0]
    P_obs = [dc.CylinderMRP(0.6,3.0), dc.CapsuleMRP(0.2,5.0), dc.SphereMRP(0.8),
         dc.ConeMRP(2.0, deg2rad(22)),dc.PolytopeMRP(SMatrix{8,3}(A2),SVector{8}(b2)),dc.PolygonMRP(dc.create_n_sided(5,0.6)...,0.2),
         dc.CylinderMRP(1.1,2.3), dc.CapsuleMRP(0.8,1.0), dc.SphereMRP(0.5), P_bot, P_top]
    Random.seed!(1234)
    gr = range(-5,5,length = 9)
    for i = 1:9
        P_obs[i].r = SA[gr[i], randn() , 3 + 1.0*randn()]
        P_obs[i].p = dc.mrp_from_q(normalize((@SVector randn(4))))
    end

    # control bounds
    u_min = -2000*ones(nu)
    u_max =  2000*ones(nu)
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
        Xref = Xref,
        Uref = Uref,
        dt = dt,
        obstacle,
        obstacle_R,
        P_obs,
        P_vic
    );


    X = [deepcopy(x0) for i = 1:N]
    U = [(.0001*randn(nu) + Uref[i]) for i = 1:N-1]

    Xn = deepcopy(X)
    Un = deepcopy(U)


    P = [zeros(nx,nx) for i = 1:N]   # cost to go quadratic term
    p = [zeros(nx) for i = 1:N]      # cost to go linear term
    d = [zeros(nu) for i = 1:N-1]    # feedforward control
    K = [zeros(nu,nx) for i = 1:N-1] # feedback gain
    iLQR(params,X,U,P,p,K,d,Xn,Un;atol=1e-3,max_iters = 3000,verbose = true,ρ = 1e0, ϕ = 10.0 )

    animation = true
    if animation
        mc.setprop!(vis["/Background"], "top_color", colorant"transparent")
        mc.setvisible!(vis["/Grid"],false)
        dc.set_floor!(vis; x = 20, y = 20, darkmode = false)

        coll = Random.shuffle(range(HSVA(0,0.5,.75,1.0), stop=HSVA(-360,0.5,.75,1.0), length=9))
        #
        for i = 1:9
            name = "P" * string(i)
            dc.build_primitive!(vis, P_obs[i], name; α = 1.0,color = coll[i])
            dc.update_pose!(vis[name],P_obs[i])
        end
        for i = 10:11
            name = "P" * string(i)
            dc.build_primitive!(vis, P_obs[i], name; α = 1.0,color = mc.RGBA(.6,.6,.6, 1.0))
            dc.update_pose!(vis[name],P_obs[i])
        end

        robot_obj = mc.MeshFileGeometry(joinpath(@__DIR__,"quadrotor.obj"))
        mc.setobject!(vis[:vic], robot_obj)
        dc.vis_traj!(vis, :traj, X; R = 0.01, color = mc.RGBA(1.0, 0.0, 0.0, 1.0))
        anim = mc.Animation(floor(Int,1/dt))
        for i = 1:length(P_obs)
            name = "P" * string(i)
            mc.settransform!(vis[name], mc.Translation(P_obs[i].r) ∘ mc.LinearMap(dc.dcm_from_mrp(P_obs[i].p)))
        end
        #
        for k = 1:N
            mc.atframe(anim, k) do
                for i = 1:length(P_obs)
                    name = "P" * string(i)
                    mc.settransform!(vis[name], mc.Translation(P_obs[i].r - X[k][1:3]) ∘ mc.LinearMap(dc.dcm_from_mrp(P_obs[i].p)))
                end
                mc.settransform!(vis[:floor], mc.Translation(-X[k][1:3]))
                mc.settransform!(vis[:traj], mc.Translation(-X[k][1:3]))
                mc.settransform!(vis[:vic], mc.LinearMap(1.5*(dc.dcm_from_mrp(X[k][SA[7,8,9]]))))
            end
        end
        mc.setanimation!(vis, anim)
    else
        mc.setvisible!(vis["/Background"],false)
        mc.setvisible!(vis["/Grid"],false)
        mc.setvisible!(vis["/Axes"],false)
        coll = Random.shuffle(range(HSVA(0,0.5,.9,1.0), stop=HSVA(-340,0.5,.9,1.0), length=9)) # inverse rotation
        #
        for i = 1:9
            name = "P" * string(i)
            dc.build_primitive!(vis, P_obs[i], name; α = 1.0,color = coll[i])
            dc.update_pose!(vis[name],P_obs[i])
        end
        for i = 10:11
            name = "P" * string(i)
            dc.build_primitive!(vis, P_obs[i], name; α = 1.0,color = mc.RGBA(.6,.6,.6, 1.0))
            dc.update_pose!(vis[name],P_obs[i])
        end

        robot_obj = mc.MeshFileGeometry(joinpath(@__DIR__,"quadrotor.obj"))
        robot_mat = mc.MeshPhongMaterial(color=mc.RGBA(0.0, 0.0, 0.0, 1.0))
        dc.vis_traj!(vis, :traj, X; R = 0.02, color = mc.RGBA(1.0, 0.0, 0.0, 1.0))

        for i = [3,26,50,75,98]
            mc.setobject!(vis["vic"*string(i)], robot_obj, robot_mat)
            mc.settransform!(vis["vic"*string(i)], mc.Translation(X[i][1:3]) ∘ mc.LinearMap(1.8*(dc.dcm_from_mrp(X[i][SA[7,8,9]]))))
        end
    end
end
