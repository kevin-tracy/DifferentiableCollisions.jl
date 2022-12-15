using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), ".."))
import DifferentiableCollisions as dc
Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()

using LinearAlgebra
using StaticArrays
import ForwardDiff as FD
using Printf
using SparseArrays
import MeshCat as mc
import Random
using Colors

include(joinpath(@__DIR__,"simple_altro.jl"))

function dynamics(p::NamedTuple,x,u,k)
    r = x[1:2]
    v = x[3:4]
    θ = x[5]
    ω = x[6]

    [
    v;
    u[1:2];
    ω;
    u[3]/100
    ]
end
function discrete_dynamics(p::NamedTuple,x,u,k)
    k1 = p.dt*dynamics(p,x,        u, k)
    k2 = p.dt*dynamics(p,x + k1/2, u, k)
    k3 = p.dt*dynamics(p,x + k2/2, u, k)
    k4 = p.dt*dynamics(p,x + k3, u, k)
    x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
end

function ineq_con_x(p,x)
    p.P_vic.r = SVector{3}([x[1:2];0])
    p.P_vic.p = SVector{3}([0,0,1]*tan(x[5]/4))
    contacts= [(1 - dc.proximity(p.P_vic, p.P_obs[i])[1]) for i = 1:length(p.P_obs)]
    vcat(contacts...)
end
function ineq_con_x_jac(p,x)
    rx,ry,vx,vy,θ,ω = x
    dp_dθ = SA[0,0,1]*(1/(4*cos(θ/4)^2))
    p.P_vic.r = SVector{3}([x[1:2];0])
    p.P_vic.p = SVector{3}([0,0,1]*tan(x[5]/4))
    Js = [dc.proximity_gradient(p.P_vic, p.P_obs[i])[2] for i = 1:length(p.P_obs)]
    contact_J = [[-Js[i][1:2]' 0 0 -Js[i][4:6]'*dp_dθ 0] for i = 1:3]
    vcat(contact_J...)
end
function ineq_con_u(p,u)
    [u-p.u_max;-u + p.u_min]
end
function ineq_con_u_jac(params,u)
    nu = params.nu
    Array(float([I(nu);-I(nu)]))
end
let
    nx = 6
    nu = 3
    N = 80
    dt = 0.1
    x0 = [1.5,1.5,0,0,0,0]
    xg = [3.5,3.7,0,0,deg2rad(90),0]
    Xref = [copy(xg) for i = 1:N]
    Uref = [zeros(nu) for i = 1:N]
    Q = Diagonal(ones(nx))
    Qf = Diagonal(ones(nx))
    R = 1*Diagonal([1,1,.001])

    P_vic = dc.create_rect_prism(2.5, 0.15, 0.01)[1]

    P_obs = [dc.create_rect_prism(3.0, 3.0,  1.0)[1],
             dc.create_rect_prism(4.0, 1.0, 1.0)[1],
             dc.create_rect_prism(1.0,  5.0, 1.1)[1]]

    P_obs[1].r = SA[1.5,3.5,0.0]
    P_obs[2].r = SA[2,0.5,0]
    P_obs[3].r = SA[4.5,2.5,0]

    u_min = -200*ones(nu)
    u_max =  200*ones(nu)

    x_min = -200*ones(nx)
    x_max =  200*ones(nx)



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
    Xhist = iLQR(params,X,U,P,p,K,d,Xn,Un;atol=4e-2,max_iters = 3000,verbose = true,ρ = 1e0, ϕ = 10.0 )

    vis = mc.Visualizer()
    mc.open(vis)
    # mc.setvisible!(vis["/Background"],false)
    mc.setprop!(vis["/Background"], "top_color", mc.RGB(1,1,1))
    mc.setprop!(vis["/Background"], "bottom_color", mc.RGB(1,1,1))
    mc.setvisible!(vis["/Axes"],false)
    mc.setvisible!(vis["/Grid"],false)
    wall_w = 0.2
    P_obs = [dc.create_rect_prism(4.0 + wall_w, wall_w, .01)[1],
             dc.create_rect_prism(3.0, wall_w, .01)[1],
             dc.create_rect_prism(wall_w, 3.0, .01)[1],
             dc.create_rect_prism(wall_w, 4.0, .01)[1]]

    P_obs[1].r = SA[2.0+wall_w/2,1,0.0] - [0,wall_w/2,0]
    P_obs[2].r = SA[1.5,2.0,0] + [0,wall_w/2,0]
    P_obs[3].r = SA[3,3.5,0] - [wall_w/2,0,0]
    P_obs[4].r = SA[4,3.0,0] +[wall_w/2,0,0]
    # dc.set_floor!(vis; darkmode = false)

    for i = 1:4
        dc.build_primitive!(vis, P_obs[i], Symbol("P"*string(i)); α = 1.0,color = mc.RGBA(0,0,0,1.0))
        dc.update_pose!(vis[Symbol("P"*string(i))],P_obs[i])
    end
    dc.build_primitive!(vis, P_vic, :vic; α = 1.0,color = mc.RGBA(1,0,0,1.0))
    anim = mc.Animation(floor(Int,1/dt))
    for k = 1:N
        mc.atframe(anim, k) do
            mc.settransform!(vis[:vic], mc.Translation([X[k][1:2];0]) ∘ mc.LinearMap(dc.dcm_from_mrp(SA[0,0,1]*tan(X[k][5]/4))))
        end
    end
    mc.setanimation!(vis, anim)
end
