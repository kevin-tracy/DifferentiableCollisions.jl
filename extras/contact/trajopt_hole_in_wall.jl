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
# using MATLAB
# import DifferentialProximity as dp
import Random
using Colors


function dynamics(p::NamedTuple,x,u,k)
    [x[4:6];u]
end
function discrete_dynamics(p::NamedTuple,x,u,k)
    k1 = p.dt*dynamics(p,x,        u, k)
    k2 = p.dt*dynamics(p,x + k1/2, u, k)
    x + k2
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
        F = cholesky(Symmetric(Guu + reg*I))
        d[k] = F\gu
        K[k] = F\Gux

        # Cost-to-go Recurrence
        p[k] = gx - K[k]'*gu + K[k]'*Guu*d[k] - Gux'*d[k]
        P[k] = Gxx + K[k]'*Guu*K[k] - Gux'*K[k] - K[k]'*Gux
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
    contacts= [(1 - dc.proximity(p.P_vic, p.P_obs[i])[1]) for i = 1:length(p.P_obs)]
    # [1 - dc.proximity(p.P_vic, p.P_obs[1])[1]]
    vcat(contacts...)
end
function ineq_con_x_jac(p,x)
    # FD2.finite_difference_jacobian(_x -> ineq_con_x(p,_x),x)
    p.P_vic.r = SVector{3}(x[1:3])
    # J = [-reshape(dc.proximity_jacobian(p.P_vic, p.P_obs[1])[3][4,1:3],1,3) zeros(1,3)]
    contact_J = [ [-reshape(dc.proximity_jacobian(p.P_vic, p.P_obs[i])[3][4,1:3],1,3) zeros(1,3)] for i = 1:length(p.P_obs)]
    # @show size(J)
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

function iLQR(params,X,U,P,p,K,d,Xn,Un;atol=1e-5,max_iters = 25,verbose = true,ρ=1,ϕ=10)

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

    λ = zeros(6)

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
    @assert positions[1] ≈ x0[1:3]
    @assert positions[N] ≈ xg[1:3]
    velocities = [Δp/(N*dt) for i = 1:N]
    [[positions[i];velocities[i]] for i = 1:N]
end

function vis_traj!(vis, name, X; R = 0.1, color = mc.RGBA(1.0, 0.0, 0.0, 1.0))
    for i = 1:(length(X)-1)
        a = X[i][1:3]
        b = X[i+1][1:3]
        cyl = mc.Cylinder(mc.Point(a...), mc.Point(b...), R)
        mc.setobject!(vis[name]["p"*string(i)], cyl, mc.MeshPhongMaterial(color=color))
    end
end
# vis = mc.Visualizer()
# mc.open(vis)
let
    nx = 6
    nu = 3
    N = 60
    dt = 0.1
    x0 = [-4,-7,7,0,0,0.0]
    xg = [-4.5,7,3,0,0,0.0]
    Xref = linear_interp(dt,x0,xg,N)
    Uref = [zeros(nu) for i = 1:N]
    Q = Diagonal(ones(nx))
    Qf = Diagonal(ones(nx))
    R = Diagonal(ones(nu))
    obstacle = zeros(3)
    obstacle_R = 2.0

    # P_vic = dc.Cone(2.0, deg2rad(22))
    P_vic = dc.Sphere(0.5)

    # P_obs = [dc.Sphere(1.6), dc.Capsule(1.0,3.0), dc.Cylinder(0.8,4.0), dc.Cone()]
    function create_n_sided(N,d)
        ns = [ [cos(θ);sin(θ)] for θ = 0:(2*π/N):(2*π*(N-1)/N)]
        A = vcat(transpose.((ns))...)
        b = d*ones(N)
        return SMatrix{N,2}(A), SVector{N}(b)
    end
    # @load "/Users/kevintracy/.julia/dev/DifferentialProximity/extras/polyhedra_plotting/polytopes.jld2"
    # P_obs = [dc.Cylinder(0.6,3.0), dc.Capsule(0.2,5.0), dc.Sphere(0.8),
         # dc.Cone(2.0, deg2rad(22)),dc.Polytope(SMatrix{8,3}(A2),SVector{8}(b2)),dc.Polygon(create_n_sided(5,0.6)...,0.2),
         # dc.Cylinder(1.1,2.3), dc.Capsule(0.8,1.0), dc.Sphere(0.5),
              # dc.Cone(3.0, deg2rad(18)),dc.Polytope(SMatrix{14,3}(A1),SVector{14}(b1)),dc.Polygon(create_n_sided(8,0.8)...,0.15),
              # dc.Cylinder(0.6,3.0), dc.Capsule(0.2,5.0), dc.Sphere(0.8),
                   # dc.Cone(2.0, deg2rad(22)),dc.Polytope(SMatrix{8,3}(A2),SVector{8}(b2)),dc.Polygon(create_n_sided(5,0.6)...,0.2),
                   # dc.Cylinder(1.1,2.3), dc.Capsule(0.8,1.0), dc.Sphere(0.5),
                        # dc.Cone(3.0, deg2rad(18)),dc.Polytope(SMatrix{14,3}(A1),SVector{14}(b1)),dc.Polygon(create_n_sided(8,0.8)...,0.15), dc.Cone(3.0, deg2rad(18))]
    # P_obs[2].r = SA[0,3.0,0.0]
    # P_obs = [P_obs...,P_obs...]
    P_obs = [create_rect_prism(;len = 10.0, wid = 10.0, hei = 1.0)[1],
             create_rect_prism(;len = 10.0, wid = 10.0, hei = 1.0)[1],
             create_rect_prism(;len = 4.1, wid = 4.1, hei = 1.1)[1],
             create_rect_prism(;len = 4.1, wid = 4.1, hei = 1.1)[1]]

    P_obs[1].r = SA[-6,0,5.0]
    P_obs[1].q = SA[cos(pi/4),sin(pi/4),0,0]
    P_obs[2].r = SA[6,0,5.0]
    P_obs[2].q = SA[cos(pi/4),sin(pi/4),0,0]
    P_obs[3].r = SA[0,0,2.0]
    P_obs[3].q = SA[cos(pi/4),sin(pi/4),0,0]
    P_obs[4].r = SA[0,0,8.0]
    P_obs[4].q = SA[cos(pi/4),sin(pi/4),0,0]

    # error()
    u_min = -20*ones(nu)
    u_max =  20*ones(nu)

    # state is x y v θ
    x_min = -20*ones(nx)
    x_max =  20*ones(nx)



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
        obstacle,
        obstacle_R,
        P_obs,
        P_vic
    );


    X = [deepcopy(x0) for i = 1:N]
    U = [.0001*randn(nu) for i = 1:N-1]

    Xn = deepcopy(X)
    Un = deepcopy(U)


    P = [zeros(nx,nx) for i = 1:N]   # cost to go quadratic term
    p = [zeros(nx) for i = 1:N]      # cost to go linear term
    d = [zeros(nu) for i = 1:N-1]    # feedforward control
    K = [zeros(nu,nx) for i = 1:N-1] # feedback gain
    iLQR(params,X,U,P,p,K,d,Xn,Un;atol=1e-3,max_iters = 3000,verbose = true,ρ = 1e-2, ϕ = 10.0 )

    sph_p1 = mc.HyperSphere(mc.Point(0,0,0.0), 0.3)
    mc.setobject!(vis[:start], sph_p1, mc.MeshPhongMaterial(color = mc.RGBA(0.0,1.0,0,1.0)))
    # mc.setobject!(vis[:vic], sph_p1, mc.MeshPhongMaterial(color = mc.RGBA(0.0,0.0,1.0,1.0)))
    mc.setobject!(vis[:stop], sph_p1, mc.MeshPhongMaterial(color = mc.RGBA(1.0,0,0,1.0)))
    mc.settransform!(vis[:start], mc.Translation(x0[1:3]))
    mc.settransform!(vis[:stop], mc.Translation(xg[1:3]))
    mc.setprop!(vis["/Background"], "top_color", colorant"transparent")
    mc.setvisible!(vis["/Grid"],false)
    dc.set_floor!(vis; x = 40, y = 40)


    # @show length(P_obs)
    for i = 1:length(P_obs)
        dc.build_primitive!(vis, P_obs[i], Symbol("P"*string(i)); α = 1.0,color = mc.RGBA(normalize(abs.(randn(3)))..., 1.0))
        dc.update_pose!(vis[Symbol("P"*string(i))],P_obs[i])
    end

    # for i = 1:length(grid_xy)
    # for i = 1:length(P_obs)
    #     sph_p1 = mc.HyperSphere(mc.Point(grid_xy[i]...,4.0), 0.3)
    #     mc.setobject!(vis["s"*string(i)], sph_p1, mc.MeshPhongMaterial(color = mc.RGBA(1.0,0,0,1.0)))
    # end

    # build actual vehicle
    dc.build_primitive!(vis, P_vic, :vic; α = 1.0,color = mc.RGBA(normalize(abs.(randn(3)))..., 1.0))

    # # visualize trajectory
    vis_traj!(vis, :traj, X; R = 0.1, color = mc.RGBA(1.0, 0.0, 0.0, 1.0))
    #
    anim = mc.Animation(floor(Int,1/dt))
    for k = 1:N
        mc.atframe(anim, k) do
            mc.settransform!(vis[:vic], mc.Translation(X[k][1:3]))
            # dc.update_pose!(vis[:vic], P_vic)
        end
    end
    mc.setanimation!(vis, anim)
end
