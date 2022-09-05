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
using JLD2
using Interpolations
# using MATLAB
# import DifferentialProximity as dp
import Random
using Colors


function skew(ω::Vector{T}) where {T}
    return [0 -ω[3] ω[2];
            ω[3] 0 -ω[1];
            -ω[2] ω[1] 0]
end
function dynamics(params::NamedTuple,x,u,k_iter)
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
function stage_cost(p::NamedTuple,x,u,k)
    dx = x - p.Xref[k]
    du = u - p.Uref[k]
    return 0.5*dx'*p.Q*dx + 0.5*du'*p.R*du
end
function term_cost(p::NamedTuple,x)
    dx = x - p.Xref[p.N]
    return 0.5*dx'*p.Qf*dx
end
function stage_cost_expansion(p::NamedTuple,x,u,k)
    dx = x - p.Xref[k]
    du = u - p.Uref[k]
    return p.Q, p.Q*dx, p.R, p.R*du
end
function term_cost_expansion(p::NamedTuple,x)
    dx = x - p.Xref[p.N]
    return p.Qf, p.Qf*dx
end
function backward_pass!(params,X,U,P,p,d,K,reg,μ,μx,ρ,λ)

    N = params.N
    ΔJ = 0.0

    # terminal cost expansion
    P[N], p[N] = term_cost_expansion(params,X[N])

    # add AL for x cons
    hxv = ineq_con_x(params,X[N])
    mask = eval_mask(μx[N],hxv)
    ∇hx = ineq_con_x_jac(params,X[N])

    p[N]  += ∇hx'*(μx[N] + ρ*(mask * hxv))
    P[N]  += ρ*∇hx'*mask*∇hx

    # add goal constraint
    hxv = X[N] - params.Xref[N]
    ∇hx = diagm(ones(params.nx))

    p[N]  += ∇hx'*(λ + ρ*hxv)
    P[N]  += ρ*∇hx'∇hx

    for k = (N-1):(-1):1

        # dynamics jacobians
        A = FD.jacobian(_x -> discrete_dynamics(params,_x,U[k],k),X[k])
        B = FD.jacobian(_u -> discrete_dynamics(params,X[k],_u,k),U[k])

        # cost expansion
        Jxx,Jx,Juu,Ju = stage_cost_expansion(params,X[k],U[k],k)

        # control constraints
        huv = ineq_con_u(params,U[k])
        mask = eval_mask(μ[k],huv)
        ∇hu = ineq_con_u_jac(params,U[k])
        Ju  += ∇hu'*(μ[k] + ρ*(mask * huv))
        Juu += ρ*∇hu'*mask*∇hu

        # state constraints
        hxv = ineq_con_x(params,X[k])
        mask = eval_mask(μx[k],hxv)
        ∇hx = ineq_con_x_jac(params,X[k])
        Jx  += ∇hx'*(μx[k] + ρ*(mask * hxv))
        Jxx += ρ*∇hx'*mask*∇hx

        # Q expansion
        gx = Jx + A'*p[k+1]
        gu = Ju + B'*p[k+1]

        Gxx = Jxx + A'*P[k+1]*A
        Guu = Juu + B'*P[k+1]*B
        Gux = B'*P[k+1]*A

        # Calculate Gains
        # @show norm(Guu - Guu')
        # @show eigvals(Guu)
        F = cholesky(Symmetric(Guu + reg*I))
        d[k] = F\gu
        K[k] = F\Gux

        # Cost-to-go Recurrence
        # p[k] = gx - K[k]'*gu + K[k]'*Guu*d[k] - Gux'*d[k]
        # P[k] = Gxx + K[k]'*Guu*K[k] - Gux'*K[k] - K[k]'*Gux

        P[k] = Jxx + K[k]'*Juu*K[k] + (A-B*K[k])'*P[k+1]*(A-B*K[k])
        p[k] = Jx - K[k]'*Ju + K[k]'*Juu*d[k] + (A - B*K[k])'*(p[k+1] - P[k+1]*B*d[k])
        ΔJ += gu'*d[k]
    end

    return ΔJ
end
function trajectory_AL_cost(params,X,U,μ,μx,ρ,λ)
    N = params.N
    J = 0.0
    for k = 1:N-1
        J += stage_cost(params,X[k],U[k],k)

        # AL terms
        huv = ineq_con_u(params,U[k])
        mask = eval_mask(μ[k],huv)
        J += dot(μ[k],huv) + 0.5*ρ*huv'*mask*huv

        hxv = ineq_con_x(params,X[k])
        mask = eval_mask(μx[k],hxv)
        J += dot(μx[k],hxv) + 0.5*ρ*hxv'*mask*hxv
    end
    J += term_cost(params,X[N])
    hxv = ineq_con_x(params,X[params.N])
    mask = eval_mask(μx[params.N],hxv)
    J += dot(μx[params.N],hxv) + 0.5*ρ*hxv'*mask*hxv

    # goal constraint
    hxv = X[N] - params.Xref[N]
    J += dot(λ,hxv) + 0.5*ρ*hxv'*hxv
    return J
end
function forward_pass!(params,X,U,K,d,ΔJ,Xn,Un,μ,μx,ρ,λ; max_linesearch_iters = 20)

    N = params.N
    α = 1.0

    J = trajectory_AL_cost(params,X,U,μ,μx,ρ,λ)
    for i = 1:max_linesearch_iters

        # Forward Rollout
        for k = 1:(N-1)
            Un[k] = U[k] - α*d[k] - K[k]*(Xn[k]-X[k])
            Xn[k+1] = discrete_dynamics(params,Xn[k],Un[k],k)
        end
        Jn = trajectory_AL_cost(params,Xn,Un,μ,μx,ρ,λ)

        # armijo line search
        if Jn < J
            X .= Xn
            U .= Un
            return Jn, α
        else
            α *= 0.5
        end
    end

    @warn "forward pass failed, adding regularization"
    α = 0.0
    return J, α
end
function update_reg(reg,reg_min,reg_max,α)
    if α == 0.0
        if reg == reg_max
            error("reached max reg")
        end
        return min(reg_max,reg*10)
    end
    if α == 1.0
        return max(reg_min,reg/10)
    end
    return reg
end
function calc_max_d(d)
    dm = 0.0
    for i = 1:length(d)
        dm = max(dm,norm(d[i]))
    end
    return dm
end
function ineq_con_u(p,u)
    [u-p.u_max;-u + p.u_min]
end
function ineq_con_u_jac(params,u)
    nu = params.nu
    Array(float([I(nu);-I(nu)]))
end
function ineq_con_x(p,x)
    # [x-p.x_max;-x + p.x_min]
    # [p.obstacle_R^2 - norm(x[1:3] - p.obstacle)^2]
    p.P_vic.r = SVector{3}(x[1:3])
    p.P_vic.p = SVector{3}(x[7:9])
    contacts= [(1 - dc.proximity(p.P_vic, p.P_obs[i])[1]) for i = 1:length(p.P_obs)]
    # [1 - dc.proximity(p.P_vic, p.P_obs[1])[1]]
    vcat(contacts...)
end
function ineq_con_x_jac(p,x)
    # FD2.finite_difference_jacobian(_x -> ineq_con_x(p,_x),x)
    p.P_vic.r = SVector{3}(x[1:3])
    p.P_vic.p = SVector{3}(x[7:9])
    # # J = [-reshape(dc.proximity_jacobian(p.P_vic, p.P_obs[1])[3][4,1:3],1,3) zeros(1,3)]
    contact_J = [ [-reshape(dc.proximity_jacobian(p.P_vic, p.P_obs[i])[3][4,1:3],1,3) zeros(1,3) -reshape(dc.proximity_jacobian(p.P_vic, p.P_obs[i])[3][4,4:6],1,3) zeros(1,3)] for i = 1:length(p.P_obs)]
    # # # @show size(J)
    vcat(contact_J...)
end
function eval_mask(μv,huv)
    # active set mask
    mask = Diagonal(zeros(length(huv)))
    for i = 1:length(huv)
        mask[i,i] = μv[i] > 0 || huv[i] > 0
    end
    mask
end

function iLQR(params,X,U,P,p,K,d,Xn,Un;atol=1e-5,max_iters = 25,verbose = true,ρ=1e0,ϕ=10)

    # # inital logging stuff
    # if verbose
    #     @printf "iter     J           ΔJ        |d|         α        reg         ρ\n"
    #     @printf "---------------------------------------------------------------------\n"
    # end

    # initial rollout
    N = params.N
    for i = 1:N-1
        X[i+1] = discrete_dynamics(params,X[i],U[i],i)
    end

    # @show [any(isnan.(x)) for x in X]
    # error()

    reg = 1e-6
    reg_min = 1e-6
    reg_max = 1e-1

    μ = [zeros(params.ncu) for i = 1:N-1]

    μx = [zeros(params.ncx) for i = 1:N]

    λ = zeros(12)

    for iter = 1:max_iters
        ΔJ = backward_pass!(params,X,U,P,p,d,K,reg,μ,μx,ρ,λ)
        J, α = forward_pass!(params,X,U,K,d,ΔJ,Xn,Un,μ,μx,ρ,λ)
        reg = update_reg(reg,reg_min,reg_max,α)
        dmax = calc_max_d(d)
        if verbose
            if rem(iter-1,10)==0
                @printf "iter     J           ΔJ        |d|         α        reg         ρ\n"
                @printf "---------------------------------------------------------------------\n"
            end
            @printf("%3d   %10.3e  %9.2e  %9.2e  %6.4f   %9.2e   %9.2e\n",
              iter, J, ΔJ, dmax, α, reg,ρ)
        end
        if (α > 0) & (dmax<atol)
            # check convio
            convio = 0

            # control constraints
            for k = 1:N-1
                huv = ineq_con_u(params,U[k])
                mask = eval_mask(μ[k],huv)

                # update dual
                μ[k] = max.(0,μ[k] + ρ*mask*huv)
                convio = max(convio,norm(huv + abs.(huv),Inf))
            end

            # state constraints
            for k = 1:N
                hxv = ineq_con_x(params,X[k])
                mask = eval_mask(μx[k],hxv)

                # update dual
                μx[k] = max.(0,μx[k] + ρ*mask*hxv)
                convio = max(convio,norm(hxv + abs.(hxv),Inf))
            end

            # goal constraint
            hxv = X[N] - params.Xref[N]
            λ .+= ρ*hxv
            convio = max(convio, norm(hxv,Inf))

            @show convio
            if convio <1e-4
                @info "success!"
                return nothing
            end

            ρ *= ϕ
            # ρ = min(1e6,ρ*ϕ)
        end
    end
    error("iLQR failed")
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

function vis_traj!(vis, name, X; R = 0.1, color = mc.RGBA(1.0, 0.0, 0.0, 1.0))
    for i = 1:(length(X)-1)
        a = X[i][1:3]
        b = X[i+1][1:3]
        cyl = mc.Cylinder(mc.Point(a...), mc.Point(b...), R)
        mc.setobject!(vis[name]["p"*string(i)], cyl, mc.MeshPhongMaterial(color=color))
    end
end
function create_rect_prism_quat(;len = 20.0, wid = 20.0, hei = 2.0)
# len = 20.0
# wid = 20.0
# hei = 2.0

    ns = [SA[1,0,0.0], SA[0,1,0.0], SA[0,0,1.0],SA[-1,0,0.0], SA[0,-1,0.0], SA[0,0,-1.0]]
    cs = [SA[len/2,0,0.0], SA[0,wid/2,0.0], SA[0,0,hei/2],SA[-len/2,0,0.0], SA[0,-wid/2,0.0], SA[0,0,-hei/2]]

    A = zeros(6,3)
    b = zeros(6)

    for i = 1:6
        A[i,:] = ns[i]'
        b[i] = dot(ns[i],cs[i])
    end

    A = SMatrix{6,3}(A)
    b = SVector{6}(b)

    mass = len*wid*hei

    inertia = (mass/12)*Diagonal(SA[wid^2 + hei^2, len^2 + hei^2, len^2 + wid^2])

    return dc.PolytopeMRP(A,b), mass, inertia
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

    # P_vic = dc.ConeMRP(0.5, deg2rad(22))
    P_vic = dc.SphereMRP(0.25)

    # P_obs = [dc.Sphere(1.6), dc.Capsule(1.0,3.0), dc.Cylinder(0.8,4.0), dc.Cone()]
    function create_n_sided(N,d)
        ns = [ [cos(θ);sin(θ)] for θ = 0:(2*π/N):(2*π*(N-1)/N)]
        A = vcat(transpose.((ns))...)
        b = d*ones(N)
        return SMatrix{N,2}(A), SVector{N}(b)
    end
    @load "/Users/kevintracy/.julia/dev/DifferentialProximity/extras/polyhedra_plotting/polytopes.jld2"
    # P_obs = [dc.Cylinder(0.6,3.0), dc.Capsule(0.2,5.0), dc.Sphere(0.8),
    #      dc.Cone(2.0, deg2rad(22)),dc.Polytope(SMatrix{8,3}(A2),SVector{8}(b2)),dc.Polygon(create_n_sided(5,0.6)...,0.2),
    #      dc.Cylinder(1.1,2.3), dc.Capsule(0.8,1.0), dc.Sphere(0.5)]
    #           dc.Cone(3.0, deg2rad(18)),dc.Polytope(SMatrix{14,3}(A1),SVector{14}(b1)),dc.Polygon(create_n_sided(8,0.8)...,0.15),
    #           dc.Cylinder(0.6,3.0), dc.Capsule(0.2,5.0), dc.Sphere(0.8),
    #                dc.Cone(2.0, deg2rad(22)),dc.Polytope(SMatrix{8,3}(A2),SVector{8}(b2)),dc.Polygon(create_n_sided(5,0.6)...,0.2),
    #                dc.Cylinder(1.1,2.3), dc.Capsule(0.8,1.0), dc.Sphere(0.5),
    #                      dc.Cone(3.0, deg2rad(18)),dc.Polytope(SMatrix{14,3}(A1),SVector{14}(b1)),dc.Polygon(create_n_sided(8,0.8)...,0.15), dc.Cone(3.0, deg2rad(18))]

    P_bot = create_rect_prism_quat(;len = 20, wid = 5.0, hei = 0.2)[1]
    P_bot.r = [0,0,0.9]
    P_top = create_rect_prism_quat(;len = 20, wid = 5.0, hei = 0.2)[1]
    P_top.r = [0,0,6.0]
    P_obs = [dc.CylinderMRP(0.6,3.0), dc.CapsuleMRP(0.2,5.0), dc.SphereMRP(0.8),
         dc.ConeMRP(2.0, deg2rad(22)),dc.PolytopeMRP(SMatrix{8,3}(A2),SVector{8}(b2)),dc.PolygonMRP(create_n_sided(5,0.6)...,0.2),
         dc.CylinderMRP(1.1,2.3), dc.CapsuleMRP(0.8,1.0), dc.SphereMRP(0.5), P_bot, P_top]
    # P_obs = [P_obs...,P_bot, P_top]
    using Random
    # @show length(P_obs)
    Random.seed!(1234)
    gr = range(-5,5,length = 9)
    # grid_xy = vec([SA[i,j] for i = gr, j = gr])
    for i = 1:9
        P_obs[i].r = SA[gr[i], randn() , 3 + 1.0*randn()]
        P_obs[i].p = dc.mrp_from_q(normalize((@SVector randn(4))))
    end
    # error()
    u_min = -2000*ones(nu)
    u_max =  2000*ones(nu)

    # state is x y v θ
    x_min = -2000*ones(nx)
    x_max =  2000*ones(nx)



    ncx = length(P_obs)
    # ncx = 2*nx
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
        obstacle,
        obstacle_R,
        P_obs,
        P_vic
    );


    X = [deepcopy(x0) for i = 1:N]
    U = [(.0001*randn(nu) + Uref[i]) for i = 1:N-1]

    # for i = 1:N-1
    #     X[i+1] = discrete_dynamics(params,X[i],U[i],i)
    # end

    Xn = deepcopy(X)
    Un = deepcopy(U)


    P = [zeros(nx,nx) for i = 1:N]   # cost to go quadratic term
    p = [zeros(nx) for i = 1:N]      # cost to go linear term
    d = [zeros(nu) for i = 1:N-1]    # feedforward control
    K = [zeros(nu,nx) for i = 1:N-1] # feedback gain
    iLQR(params,X,U,P,p,K,d,Xn,Un;atol=1e-3,max_iters = 3000,verbose = true,ρ = 1e-2, ϕ = 10.0 )

    animation = false
    if animation
        # sph_p1 = mc.HyperSphere(mc.Point(0,0,0.0), 0.1)
        mc.setprop!(vis["/Background"], "top_color", colorant"transparent")
        mc.setvisible!(vis["/Grid"],false)
        dc.set_floor!(vis; x = 20, y = 20, darkmode = false)
        #
        #
        # # @show length(P_obs)
        coll = shuffle(range(HSVA(0,0.5,.75,1.0), stop=HSVA(-360,0.5,.75,1.0), length=9))
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

        robot_obj = mc.MeshFileGeometry("/Users/kevintracy/.julia/dev/DCOL/extras/paper_vis/quadrotor.obj")
        mc.setobject!(vis[:vic], robot_obj)
        vis_traj!(vis, :traj, X; R = 0.01, color = mc.RGBA(1.0, 0.0, 0.0, 1.0))
        anim = mc.Animation(floor(Int,1/dt))
        for i = 1:length(P_obs)
            name = "P" * string(i)
            mc.settransform!(vis[name], mc.Translation(P_obs[i].r) ∘ mc.LinearMap(dc.dcm_from_mrp(P_obs[i].p)))
        end
        #
        for k = 1:N
            mc.atframe(anim, k) do
                mc.settransform!(vis["/Cameras/default"], mc.Translation([-4.0,1.5,-0.5]))
                for i = 1:length(P_obs)
                    name = "P" * string(i)
                    mc.settransform!(vis[name], mc.Translation(P_obs[i].r - X[k][1:3]) ∘ mc.LinearMap(dc.dcm_from_mrp(P_obs[i].p)))
                #
                end
                mc.settransform!(vis[:floor], mc.Translation(-X[k][1:3]))
                mc.settransform!(vis[:traj], mc.Translation(-X[k][1:3]))
                mc.settransform!(vis[:vic], mc.LinearMap(1.5*(dc.dcm_from_mrp(X[k][SA[7,8,9]]))))
                # mc.settransform!(vis[:vic], mc.Translation(X[k][1:3]))
            end
        end
        mc.setanimation!(vis, anim)
    else
        # sph_p1 = mc.HyperSphere(mc.Point(0,0,0.0), 0.1)
        # mc.setprop!(vis["/Background"], "top_color", colorant"transparent")
        mc.setvisible!(vis["/Background"],false)
        mc.setvisible!(vis["/Grid"],false)
        mc.setvisible!(vis["/Axes"],false)
        # dc.set_floor!(vis; x = 20, y = 20, darkmode = false)
        #
        #
        # # @show length(P_obs)
        coll = shuffle(range(HSVA(0,0.5,.9,1.0), stop=HSVA(-340,0.5,.9,1.0), length=9)) # inverse rotation
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

        robot_obj = mc.MeshFileGeometry("/Users/kevintracy/.julia/dev/DCOL/extras/paper_vis/quadrotor.obj")
        robot_mat = mc.MeshPhongMaterial(color=mc.RGBA(0.0, 0.0, 0.0, 1.0))
        # mc.setobject!(vis[:vic], robot_obj)
        vis_traj!(vis, :traj, X; R = 0.02, color = mc.RGBA(1.0, 0.0, 0.0, 1.0))

        for i = [3,26,50,75,98]
            mc.setobject!(vis["vic"*string(i)], robot_obj, robot_mat)
            mc.settransform!(vis["vic"*string(i)], mc.Translation(X[i][1:3]) ∘ mc.LinearMap(1.8*(dc.dcm_from_mrp(X[i][SA[7,8,9]]))))
        end
        # for i = 1:length(P_obs)
        #     name = "P" * string(i)
        #     mc.settransform!(vis[name], mc.Translation(P_obs[i].r) ∘ mc.LinearMap(dc.dcm_from_mrp(P_obs[i].p)))
        # end
        #
        # for k = 1:N
        #     mc.atframe(anim, k) do
        #         mc.settransform!(vis["/Cameras/default"], mc.Translation([-4.0,1.5,-0.5]))
        #         for i = 1:length(P_obs)
        #             name = "P" * string(i)
        #             mc.settransform!(vis[name], mc.Translation(P_obs[i].r - X[k][1:3]) ∘ mc.LinearMap(dc.dcm_from_mrp(P_obs[i].p)))
        #         #
        #         end
        #         mc.settransform!(vis[:floor], mc.Translation(-X[k][1:3]))
        #         mc.settransform!(vis[:traj], mc.Translation(-X[k][1:3]))
        #         mc.settransform!(vis[:vic], mc.LinearMap(1.5*(dc.dcm_from_mrp(X[k][SA[7,8,9]]))))
        #         # mc.settransform!(vis[:vic], mc.Translation(X[k][1:3]))
        #     end
        # end
        # mc.setanimation!(vis, anim)
    end
end
