# using Pkg
# Pkg.activate(joinpath(@__DIR__, ".."))
# using DCOL
# Pkg.activate(@__DIR__)
# Pkg.instantiate()

# using LinearAlgebra
# using Printf
# import ForwardDiff as FD

# -------------------THIS IS ALL ALTRO-------------------------------
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
    # backwards pass for Altro
    # P - vector of cost to go quadratic terms (matrices)
    # p - vector of cost to go linear terms (vectors)
    # K - vector of feedback gain matrices (matrices)
    # d - vector of feedforward controls (vectors)


    N = params.N
    ΔJ = 0.0

    # terminal cost expansion
    P[N], p[N] = term_cost_expansion(params,X[N])

    # add AL terms for the state constraint at the final time step
    hxv = ineq_con_x(params,X[N])
    mask = eval_mask(μx[N],hxv)
    ∇hx = ineq_con_x_jac(params,X[N])

    # add these into the CTG p and P
    p[N]  += ∇hx'*(μx[N] + ρ*(mask * hxv))
    P[N]  += ρ*∇hx'*mask*∇hx

    # add AL terms for goal constraint
    hxv = X[N] - params.Xref[N]
    ∇hx = diagm(ones(params.nx))

    # add these into the CTG p and P
    p[N]  += ∇hx'*(λ + ρ*hxv)
    P[N]  += ρ*∇hx'∇hx

    # iterate from N-1 to 1 backwards
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

        # # un regularized
        # Gxx = Jxx + A'*P[k+1]*A
        # Guu = Juu + B'*P[k+1]*B
        # Gux = B'*P[k+1]*A

        # regularized
        Gxx = Jxx + A'*(P[k+1] + reg*I)*A
        Guu = Juu + B'*(P[k+1] + reg*I)*B
        Gux = B'*(P[k+1] + reg*I)*A

        # Calculate Gains
        F = cholesky(Symmetric(Guu))
        d[k] = F\gu
        K[k] = F\Gux

        # Cost-to-go Recurrence
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

        # AL terms for ineq_con_u
        huv = ineq_con_u(params,U[k])
        mask = eval_mask(μ[k],huv)
        J += dot(μ[k],huv) + 0.5*ρ*huv'*mask*huv

        # AL terms for ineq_con_x
        hxv = ineq_con_x(params,X[k])
        mask = eval_mask(μx[k],hxv)
        J += dot(μx[k],hxv) + 0.5*ρ*hxv'*mask*hxv
    end

    # AL terms for state constraint at last time step
    J += term_cost(params,X[N])
    hxv = ineq_con_x(params,X[params.N])
    mask = eval_mask(μx[params.N],hxv)
    J += dot(μx[params.N],hxv) + 0.5*ρ*hxv'*mask*hxv

    # AL terms for goal constraint
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

        # linesearch
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
function eval_mask(μv,huv)
    # active set mask
    mask = Diagonal(zeros(length(huv)))
    for i = 1:length(huv)
        mask[i,i] = μv[i] > 0 || huv[i] > 0
    end
    mask
end

function iLQR(params,X,U,P,p,K,d,Xn,Un;atol=1e-3,max_iters = 250,verbose = true,ρ=1,ϕ=10)

    # first check the sizes of everything
    @assert length(X) == params.N
    @assert length(U) == params.N-1
    @assert length(X[1]) == params.nx
    @assert length(U[1]) == params.nu
    @assert length(ineq_con_u(params,U[1])) == params.ncu
    @assert length(ineq_con_x(params,X[1])) == params.ncx

    # keep track of trajectories for each iterate
    Xhist=[deepcopy(X) for i = 1:1000]

    # initial rollout
    N = params.N
    for i = 1:N-1
        X[i+1] = discrete_dynamics(params,X[i],U[i],i)
    end

    Xhist[1] .= X


    reg = 1e-6
    reg_min = 1e-6
    reg_max = 1e2

    μ = [zeros(params.ncu) for i = 1:N-1]

    μx = [zeros(params.ncx) for i = 1:N]

    λ = zeros(params.nx)

    for iter = 1:max_iters
        ΔJ = backward_pass!(params,X,U,P,p,d,K,reg,μ,μx,ρ,λ)
        J, α = forward_pass!(params,X,U,K,d,ΔJ,Xn,Un,μ,μx,ρ,λ)

        Xhist[iter + 1] .= X

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
                return Xhist[1:(iter + 1)]
            end

            ρ *= ϕ
        end
    end
    error("iLQR failed")
end
#----------------------ALTRO DONE --------------------------

#---------------------THIS IS WHAT YOU NEED TO INPUT--------
# function dynamics(p::NamedTuple,x,u,k)
#     # dynamis for a cart pole
#     mc = p.mc
#     mp = p.mp
#     l = p.l
#     g = p.g
#
#     q = x[1:2]
#     qd = x[3:4]
#
#     s = sin(q[2])
#     c = cos(q[2])
#
#     H = [mc+mp mp*l*c; mp*l*c mp*l^2]
#     C = [0 -mp*qd[2]*l*s; 0 0]
#     G = [0, mp*g*l*s]
#     B = [1, 0]
#
#     qdd = -H\(C*qd + G - B*u[1])
#     return [qd; qdd]
# end
# function discrete_dynamics(p::NamedTuple,x,u,k)
#     # RK4
#     k1 = p.dt*dynamics(p,x,        u, k)
#     k2 = p.dt*dynamics(p,x + k1/2, u, k)
#     k3 = p.dt*dynamics(p,x + k2/2, u, k)
#     k4 = p.dt*dynamics(p,x + k3, u, k)
#     x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
# end
# function ineq_con_x(p,x)
#     [x-p.x_max;-x + p.x_min]
# end
# function ineq_con_u(p,u)
#     [u-p.u_max;-u + p.u_min]
# end
# function ineq_con_u_jac(params,u)
#     FD.jacobian(_u -> ineq_con_u(params,_u), u)
# end
# function ineq_con_x_jac(p,x)
#     FD.jacobian(_x -> ineq_con_x(p,_x),x)
# end
#
# # here is the script
# let
#     nx = 4
#     nu = 1
#     N = 50
#     dt = 0.1
#     x0 = [0,0,0,0.]
#     xg = [0,pi,0,0]
#     Xref = [deepcopy(xg) for i = 1:N]
#     Uref = [zeros(nu) for i = 1:N-1]
#     Q = 1e-2*Diagonal([1,1,1,1.0])
#     R = 1e-1*Diagonal([1.0])
#     Qf = 1*Diagonal([1,1,1,1.0])
#
#     u_min = -20*ones(nu)
#     u_max =  20*ones(nu)
#
#     # state is x y v θ
#     x_min = -20*ones(nx)
#     x_max =  20*ones(nx)
#
#     ncx = 2*nx
#     ncu = 2*nu
#
#     params = (
#         nx = nx,
#         nu = nu,
#         ncx = ncx,
#         ncu = ncu,
#         N = N,
#         Q = Q,
#         R = R,
#         Qf = Qf,
#         u_min = u_min,
#         u_max = u_max,
#         x_min = x_min,
#         x_max = x_max,
#         Xref = Xref,
#         Uref = Uref,
#         dt = dt,
#         mc = 1.0,
#         mp = 0.2,
#         l = 0.5,
#         g = 9.81,
#     );
#
#
#     X = [deepcopy(x0) for i = 1:N]
#     U = [.01*randn(nu) for i = 1:N-1]
#
#     Xn = deepcopy(X)
#     Un = deepcopy(U)
#
#
#     P = [zeros(nx,nx) for i = 1:N]   # cost to go quadratic term
#     p = [zeros(nx) for i = 1:N]      # cost to go linear term
#     d = [zeros(nu) for i = 1:N-1]    # feedforward control
#     K = [zeros(nu,nx) for i = 1:N-1] # feedback gain
#     Xhist = iLQR(params,X,U,P,p,K,d,Xn,Un;atol=1e-1,max_iters = 3000,verbose = true,ρ = 1e0, ϕ = 10.0 )
#
# end
