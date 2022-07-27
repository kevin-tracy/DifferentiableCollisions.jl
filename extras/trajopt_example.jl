using Pkg
Pkg.activate(dirname(@__FILE__))
using LinearAlgebra
import ForwardDiff as FD
using Printf
import MeshCat as mc
import FiniteDiff
using StaticArrays
import DCD
# now with attitude
function skew(ω)
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
function dynamics(p::NamedTuple, x, u)
    # particle (double integrator)
    x1 = x[1:12]
    x2 = x[13:24]
    u1 = u[1:6]
    u2 = u[7:12]
    [
        rigid_body_dynamics(p.J1,p.m1,x1,u1)
        rigid_body_dynamics(p.J2,p.m2,x2,u2)
    ]
end
function discrete_dynamics(p::NamedTuple,x,u,k)
    k1 = p.dt*dynamics(p,x,u)
    k2 = p.dt*dynamics(p,x + k1/2,u)
    k3 = p.dt*dynamics(p,x - k1 + 2*k2, u)
    x + (k1 + 4k2 + k3)/6
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
    ∇hx = I(params.nx)

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
        ∇hu = ineq_con_u_jac()
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
function ineq_con_u_jac()
    Array(float([I(12);-I(12)]))
end
function ineq_con_x(params,x)
    x1 = x[SVector{12}(1:12)]
    r1 = SVector{3}(x1[1:3])
    p1 = SVector{3}(x1[7:9])
    x2 = x[SVector{12}(13:24)]
    r2 = SVector{3}(x2[1:3])
    p2 = SVector{3}(x2[7:9])

    params.capsule.r = r1
    params.capsule.p = p1
    params.cone.r = r2
    params.cone.p = p2

    α, x = DCD.proximity(params.capsule, params.cone; pdip_tol = 1e-6)

    [1 - α^2]
end
function ineq_con_x_jac(params,x)
    # FiniteDiff.finite_difference_jacobian(_x -> ineq_con_x(params,_x), x)
    x1 = x[SVector{12}(1:12)]
    r1 = SVector{3}(x1[1:3])
    p1 = SVector{3}(x1[7:9])
    x2 = x[SVector{12}(13:24)]
    r2 = SVector{3}(x2[1:3])
    p2 = SVector{3}(x2[7:9])

    params.capsule.r = r1
    params.capsule.p = p1
    params.cone.r = r2
    params.cone.p = p2

    α, x, ∂z_∂state = DCD.proximity_jacobian(params.capsule, params.cone;pdip_tol = 1e-6)

    # @show size(∂z_∂state)
    dα_dstate = ∂z_∂state[4,:]'
    # now we do the squaring thing
    dα_dstate = 2*α*dα_dstate
    # @show size(dα_dstate)
    # error()

    dα_dr1 = dα_dstate[1,1:3]'
    dα_dp1 = dα_dstate[1,4:6]'
    dα_dr2 = dα_dstate[1,7:9]'
    dα_dp2 = dα_dstate[1,10:12]'

    -[dα_dr1 zeros(1,3) dα_dp1 zeros(1,3) dα_dr2 zeros(1,3) dα_dp2 zeros(1,3)]
end
function eval_mask(μv,huv)
    # active set mask
    mask = Diagonal(zeros(length(huv)))
    for i = 1:length(huv)
        mask[i,i] = μv[i] > 0 || huv[i] > 0
    end
    mask
end

function iLQR(params,X,U,P,p,K,d,Xn,Un;atol=1e-5,max_iters = 100,verbose = true,ρ=1,ϕ=10)

    # inital logging stuff
    if verbose
        @printf "iter     J           ΔJ        |d|         α        reg         ρ\n"
        @printf "---------------------------------------------------------------------\n"
    end

    # initial rollout
    N = params.N
    for i = 1:N-1
        X[i+1] = discrete_dynamics(params,X[i],U[i],i)
    end

    reg = 1e-6
    reg_min = 1e-8
    reg_max = 1e-1

    μ = [zeros(24) for i = 1:N-1]

    μx = [zeros(1) for i = 1:N]

    λ = zeros(params.nx)

    Xs = [deepcopy(X) for i = 1:max_iters]

    for iter = 1:max_iters
        ΔJ = backward_pass!(params,X,U,P,p,d,K,reg,μ,μx,ρ,λ)
        J, α = forward_pass!(params,X,U,K,d,ΔJ,Xn,Un,μ,μx,ρ,λ)
        Xs[iter + 1] = deepcopy(X)
        reg = update_reg(reg,reg_min,reg_max,α)
        dmax = calc_max_d(d)
        if verbose
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
                return Xs[1:(iter + 1)]
            end

            ρ *= ϕ
        end
    end
    error("iLQR failed")
end

function ics()
    r1 = [8.0,2,3]
    r2 = [12,4,4]
    v1 = zeros(3)
    v2 = zeros(3)
    p1 = normalize(randn(3))*tan(deg2rad(140)/4)
    p2 = normalize(randn(3))*tan(deg2rad(130)/4)
    ω1 = zeros(3)
    ω2 = zeros(3)
    [r1;v1;p1;ω1;r2;v2;p2;ω2]
end
function p_from_q(q)
    return q[2:4]/(1+q[1])
end
function q_from_p(p)
    return (1/(1+dot(p,p)))*[(1-dot(p,p));2*p]
end
function dcm_from_q(q)
    q4,q1,q2,q3 = normalize(q)
    [(2*q1^2+2*q4^2-1)   2*(q1*q2 - q3*q4)   2*(q1*q3 + q2*q4);
          2*(q1*q2 + q3*q4)  (2*q2^2+2*q4^2-1)   2*(q2*q3 - q1*q4);
          2*(q1*q3 - q2*q4)   2*(q2*q3 + q1*q4)  (2*q3^2+2*q4^2-1)]
end
using Random
Random.seed!(1)
let
    nx = 24
    nu = 12
    N = 60
    Q = kron(I(2),Diagonal([1,1,1,.1,.1,.1,10,10,10,.1,.1,.1]))
    Qf = 1*Q
    R = 0.02*Diagonal(ones(nu))

    u_min = -45.0*ones(nu)
    u_max =  45.0*ones(nu)


    x0 = ics()
    Xref = [[zeros(12);-2;-2;-2;zeros(9)] for i = 1:N]
    Uref = [zeros(nu) for i = 1:N-1]


    dt = 0.05

    L1 = 2.0
    R1 = 0.75
    L2 = 1.8
    R2 = 0.6



    capsule = DCD.CapsuleMRP(1.4,2.2)
    cone = DCD.ConeMRP(3.0,deg2rad(22))

    J1 = 1*Diagonal([1,2,3])
    m1 = 1.0
    J2 = 1*Diagonal([1,2,3])
    m2 = 1.0

    params = (
        nx = nx,
        nu = nu,
        N = N,
        Q = Q,
        R = R,
        Qf = Qf,
        u_min = u_min,
        u_max = u_max,
        Xref = Xref,
        Uref = Uref,
        dt = dt,
        J1 = J1,
        J2 = J2,
        m1 = m1,
        m2 = m2,
        L1 = L1,
        L2 = L2,
        R1 = R1,
        R2 = R2,
        capsule = capsule,
        cone = cone
    );



    X = [deepcopy(x0) for i = 1:N]
    U = [.0001*randn(nu) for i = 1:N-1]

    Xn = deepcopy(X)
    Un = deepcopy(U)


    P = [zeros(nx,nx) for i = 1:N]   # cost to go quadratic term
    p = [zeros(nx) for i = 1:N]      # cost to go linear term
    d = [zeros(nu) for i = 1:N-1]    # feedforward control
    K = [zeros(nu,nx) for i = 1:N-1] # feedback gain
    Xs = iLQR(params,X,U,P,p,K,d,Xn,Un;atol=1e-3,max_iters = 300,verbose = true,ρ = 1e0, ϕ = 2.0 )




    vis = mc.Visualizer()
    mc.open(vis)
    c1 = [245, 155, 66]/255
    c2 = [2,190,207]/255
    DCD.build_primitive!(vis, capsule, :capsule; α = 1.0,color = mc.RGBA(c2..., 0.7))
    DCD.build_primitive!(vis, cone, :cone; α = 1.0,color = mc.RGBA(c1..., 0.7))

    anim = mc.Animation(floor(Int,1/dt))

    for k = 1:length(X)
        mc.atframe(anim, k) do
            x1 = X[k][1:12]
            x2 = X[k][13:24]
            mc.settransform!(vis[:capsule],mc.Translation(x1[1:3]) ∘ mc.LinearMap(mc.QuatRotation(q_from_p(x1[7:9]))))
            mc.settransform!(vis[:cone],mc.Translation(x2[1:3]) ∘ mc.LinearMap(mc.QuatRotation(q_from_p(x2[7:9]))))

        end
    end
    mc.setanimation!(vis, anim)


    vis2 = mc.Visualizer()
    mc.open(vis2)
    X=Xs[20]
    c1 = [245, 155, 66]/255
    c2 = [2,190,207]/255
    DCD.build_primitive!(vis2, capsule, :capsule; α = 1.0,color = mc.RGBA(c2..., 0.7))
    DCD.build_primitive!(vis2, cone, :cone; α = 1.0,color = mc.RGBA(c1..., 0.7))

    anim = mc.Animation(floor(Int,1/dt))

    for k = 1:length(X)
        mc.atframe(anim, k) do
            x1 = X[k][1:12]
            x2 = X[k][13:24]
            mc.settransform!(vis2[:capsule],mc.Translation(x1[1:3]) ∘ mc.LinearMap(mc.QuatRotation(q_from_p(x1[7:9]))))
            mc.settransform!(vis2[:cone],mc.Translation(x2[1:3]) ∘ mc.LinearMap(mc.QuatRotation(q_from_p(x2[7:9]))))

        end
    end
    mc.setanimation!(vis2, anim)
    # mc.render(vis)


end
