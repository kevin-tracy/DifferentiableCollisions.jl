using LinearAlgebra
using Convex
using ECOS


nx = 10
nc = 4

Q = randn(nx,nx); Q = Q'*Q;
q = randn(nx)
A = randn(nc,nx); b = randn(nc)

# solve with CVX
x_cvx= Variable(nx)
prob = minimize(0.5*quadform(x_cvx,Q) + q'*x_cvx)
prob.constraints += A*x_cvx == b
solve!(prob, ECOS.Optimizer)
x_cvx = vec(x_cvx.value)

# solve with linear system
lin_sol = [Q A';A zeros(nc,nc)]\[-q;b]
x_linsol = lin_sol[1:nx]

@show norm(x_cvx - x_linsol)

# solve with augmented lagrangian
function AL(x,λ,ρ)
    0.5*x'*Q*x + q'*x + ρ*(A*x -b)'*(A*x - b)
end

ρ = 1.0
ϕ = 10.0

x = randn(nx)
λ = zeros(nc)

for i = 1:10
    H = FD.hessian( _x -> AL(_x, λ, ρ), x)
    g = FD.gradient(_x -> AL(_x, λ, ρ), x)
    @show norm(x - x_linsol)
    if norm(g)<1e-4
        λ += ρ*(A*x - b)
        ρ *= ϕ
    else
        Δx = -H\g
        # @show norm(Δx)
        α = 1.0
        x += α*Δx
    end
end
