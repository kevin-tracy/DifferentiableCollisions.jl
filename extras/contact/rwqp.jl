
using LinearAlgebra
using Convex
using ECOS


function hat(ω)
    return [0    -ω[3]  ω[2];
            ω[3]  0    -ω[1];
            -ω[2] ω[1]  0]
end

# made up problem data
Δt = 0.25                      # sample rate
b_mag = randn(3)               # earth magnetic field in sc body frame
n_wheels = 6                   # number of reaction wheels
rk = randn(n_wheels)           # current wheel speeds
r_desired = randn(n_wheels)    # desired wheel speeds
B = randn(3,n_wheels)          # matrix for mapping wheel accel to sc torque
τ_desired = randn(3)           # desired torque
γ = 3.4                        # cost tuning weight (penalty on rw)
ϕ = 135.0                      # cost tuning weight (penalty on magnetorquer)

# create and solve this optimization problem
u = Variable(n_wheels)
m = Variable(3)

#                          # wheel speed regulation     # quadratic penalties for controls
prob = minimize(sumsquares( (rk + Δt*u) - r_desired) + γ*sumsquares(u) + ϕ*sumsquares(m) )

# must produce desired torque (note there are no actuator limits here)
prob.constraints += τ_desired == B*u - hat(b_mag)*m
solve!(prob, Hypatia.Optimizer)

# now we solve this same problem in closed form
Q = diagm([(Δt^2+γ)*ones(n_wheels); ϕ*ones(3)])             # diagonal matrix
Qinv = diagm([(1/(Δt^2+γ))*ones(n_wheels); (1/ϕ)*ones(3)]) # diagonal matrix
q = [Δt*(rk - r_desired); zeros(3)]
C = [B -hat(b_mag)]
d = copy(τ_desired)

# here is how we can solve equality constrained quadratic programs
λ = -(C*Qinv*C')\(d + C*Qinv*q) # this is a 3x3 matrix, you can solve this with anything
x = Qinv*(-q - C'*λ)

@test norm(x - [u.value;m.value]) < 1e-7 # make sure it's correct within solver tolerance


# sol = [Q A';A zeros(3,3)]\[-q;b]
# @show norm(sol[1:9] - [u.value;m.value])
