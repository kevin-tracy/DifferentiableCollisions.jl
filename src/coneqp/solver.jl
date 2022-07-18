using LinearAlgebra
# import ForwardDiff as FD
using JLD2
using SparseArrays
using Printf

function ort_nt_scaling(s,z,θ)
    s_ort = s[θ.idx_ort]
    z_ort = z[θ.idx_ort]
    Diagonal(sqrt.(s_ort ./ z_ort))
end
function normalize_soc(x)
    J = Diagonal([1,-1,-1,-1])
    x̄ = x*(1/sqrt(x'*J*x))
end
function soc_nt_scaling(s,z,θ)
    s_soc = s[θ.idx_soc]
    z_soc = z[θ.idx_soc]
    J = Diagonal([1,-1,-1,-1])
    z̄ = normalize_soc(z_soc)
    s̄ = normalize_soc(s_soc)
    γ = sqrt((1 + dot(z̄,s̄))/2)
    w̄ = (1/(2*γ))*(s̄ + J*z̄)
    b = (1/(w̄[1] + 1))
    W̄ = [w̄'; w̄[2:end] (I + b*w̄[2:end]*w̄[2:end]')]
    W = W̄*((s_soc'*J*s_soc)/(z_soc'*J*z_soc))^(1/4)
end
function nt_scaling(s,z,θ)
    W1 = ort_nt_scaling(s,z,θ)
    W2 = soc_nt_scaling(s,z,θ)
    Matrix(blockdiag(sparse(W1),sparse(W2)))
end
function soc_prod(u,v)
    u0 = u[1]
    u1 = u[2:end]

    v0 = v[1]
    v1 = v[2:end]

    [dot(u,v);u0*v1 + v0*u1]
end
function cone_prod(s,z,θ)
    s_ort = s[θ.idx_ort]
    z_ort = z[θ.idx_ort]
    s_soc = s[θ.idx_soc]
    z_soc = z[θ.idx_soc]

    [s_ort .* z_ort; soc_prod(s_soc,z_soc)]
end
function arrow(v)
    [v'; v[2:end] v[1]*I]
end
function inverse_cone_prod(λ,v,θ)
    λ_ort = λ[θ.idx_ort]
    v_ort = v[θ.idx_ort]
    λ_soc = λ[θ.idx_soc]
    v_soc = v[θ.idx_soc]

    top = v_ort ./ λ_ort
    bot = arrow(λ_soc)\v_soc
    [top;bot]
end

function gen_e(ne)
    e = zeros(ne)
    e[1] = 1
    e
end

function in_soc(x)
    norm(x[2:end]) < x[1]
end

function ort_linesearch(x,dx)
    # this returns the max α ∈ [0,1] st (x + Δx > 0)
    α = 1.0
    for i = 1:length(x)
        if dx[i]<0
            α = min(α,-x[i]/dx[i])
        end
    end
    return α
end

function solve_ls(dx,dz,ds,W,λ,θ)
    bx = dx
    bz = dz - W'*(inverse_cone_prod(λ,ds,θ))

    # full way of solving
    # sol = [0*I θ.G'; θ.G -W'*W]\[bx;bz]
    # Δx = sol[θ.idx_x]
    # Δz = sol[θ.idx_s] # this mismatch is intentional

    # cholesky way of solving
    Δx = cholesky(Symmetric(θ.G'*(inv(W)^2)*θ.G))\(bx + θ.G'*inv(W)^2*bz)
    Δz = (inv(W)^2)*(θ.G*Δx - bz)

    # this is the same for both
    Δs = W'*(inverse_cone_prod(λ,ds,θ) - W*Δz)
    return Δx, Δs, Δz
end



function soc_linesearch(y,Δ)
    ν = max(y[1]^2 - dot(y[2:end],y[2:end]), 1e-25) + 1e-13
    ζ = y[1]*Δ[1] - dot(y[2:end],Δ[2:end])
    ρ = zeros(length(y))
    ρ[1] = ζ/ν
    ρ[2:end]= Δ[2:end]/sqrt(ν) - ( ( (ζ/sqrt(ν)) + Δ[1] )/( y[1]/sqrt(ν) + 1 ) )*(y[2:end]/ν)
    if norm(ρ[2:end])>ρ[1]
        return min(1, 1/(norm(ρ[2:end]) - ρ[1]))
    else
        return 1
    end
end
function linesearch(x,Δx,θ)
    x_ort  =  x[θ.idx_ort]
    Δx_ort = Δx[θ.idx_ort]
    x_soc  =  x[θ.idx_soc]
    Δx_soc = Δx[θ.idx_soc]

    α_anal =  min(ort_linesearch(x_ort,Δx_ort), soc_linesearch(x_soc, Δx_soc))
end


function build_problem()
    @load "/Users/kevintracy/.julia/dev/DCD/extras/example_socp.jld2"

    n_ort = length(h_ort)
    n_soc = length(h_soc)

    G = [G_ort;G_soc]
    h = [h_ort;h_soc]

    idx_ort = 1:n_ort
    idx_soc = (n_ort + 1):(n_ort + n_soc)

    nx = 5
    ns = n_ort + n_soc
    nz = ns

    idx_x = 1:nx
    idx_s = (nx + 1):(nx + ns)
    idx_z = (nx + ns + 1):(nx + ns + nz)


    c = [0,0,0,1,0]

    θ = (G = G, h = h, c = c, idx_ort = idx_ort, idx_soc = idx_soc,
         n_ort = n_ort, n_soc = n_soc, nx = nx, ns = ns, nz = nz,
         idx_x = idx_x, idx_s = idx_s, idx_z = idx_z)
end

function tt()

    θ = build_problem()

    x = randn(5)
    s = [ones(θ.n_ort + 1); 0.01*ones(θ.n_soc-1) ]
    z = copy(s) + 0.01*abs.(randn(length(s)))
    # x,s,z = init_coneqp(θ)

    @printf "iter     objv        gap       |Gx+s-h|      κ      step\n"
    @printf "---------------------------------------------------------\n"

    c = θ.c
    G = θ.G
    h = θ.h
    idx_x = θ.idx_x
    idx_s = θ.idx_s
    idx_z = θ.idx_z
    idx_ort = θ.idx_ort
    idx_soc = θ.idx_soc
    m = length(idx_ort) + 1

    for main_iter = 1:30

        # evaluate NT scaling
        W = nt_scaling(s,z,θ)
        λ = W*z
        λλ = cone_prod(λ,λ,θ)

        # evaluate residuals
        rx = G'*z + c
        rz = s + G*x - h
        μ = dot(s,z)/m
        if μ < 1e-4
            @info "success"
            break
        end

        # solve affine
        Δxa, Δsa, Δza =  solve_ls(-rx,-rz,-λλ,W,λ,θ)

        αa = min(linesearch(s,Δsa,θ), linesearch(z,Δza,θ))
        ρ = dot(s + αa*Δsa, z + αa*Δza)/dot(s,z)
        σ = max(0, min(1,ρ))^3

        η = 0.0
        γ = 1.0
        e = [ones(length(idx_ort)); gen_e(length(idx_soc))]
        ds = -λλ - γ*cone_prod((W')\Δsa, W*Δza,θ) + σ*μ*e
        Δx, Δs, Δz = solve_ls(-(1 - η)*rx,-(1 - η)*rz,ds,W,λ,θ)

        α = min(1,0.99*min(linesearch(s,Δs,θ), 0.99*linesearch(z,Δz,θ)))

        @printf("%3d   %10.3e  %9.2e  %9.2e  %9.2e  % 6.4f\n",
          main_iter, θ.c'*x, μ, norm(θ.G*x + s - θ.h),
          σ*μ, α)
        # @show α
        x += α*Δx
        s += α*Δs
        z += α*Δz
    end

end

tt()
