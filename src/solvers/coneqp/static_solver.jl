# cd("/Users/kevintracy/.julia/dev/DCD")
# import Pkg; Pkg.activate(".")
# using LinearAlgebra
# using StaticArrays
# using JLD2
# using BenchmarkTools
# using Printf
# import DCD

# function build_pr()
#     @load "/Users/kevintracy/.julia/dev/DCD/extras/example_socp.jld2"
#
#     nx = 5
#     n_ort = length(h_ort)
#     n_soc = length(h_soc)
#
#     G = SMatrix{n_ort + n_soc, nx}([G_ort;G_soc])
#     h = SVector{n_ort + n_soc}([h_ort;h_soc])
#
#     idx_ort = SVector{n_ort}(1:n_ort)
#     idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))
#
#     c = SA[0,0,0,1,0.0]
#
#     return c, G, h, idx_ort, idx_soc
# end

# @inline function ort_linesearch(x::SVector{n,T},dx::SVector{n,T}) where {n,T}
#     # this returns the max α ∈ [0,1] st (x + Δx > 0)
#     α = 1.0
#     for i = 1:length(x)
#         if dx[i]<0
#             α = min(α,-x[i]/dx[i])
#         end
#     end
#     return α
# end

# @inline function soc_linesearch(y::SVector{n,T},Δ::SVector{n,T}) where {n,T}
#     # TODO: maybe try cvxopt linesearch, or conelp linesearch
#     v_idx = SVector{n-1}(2:n)
#     yv = y[v_idx]
#     Δv = Δ[v_idx]
#     # ν = max(y[1]^2 - dot(yv,yv), 1e-25) + 1e-13
#     ν = y[1]^2 - dot(yv,yv)
#     ζ = y[1]*Δ[1] - dot(yv,Δv)
#     ρ = [ζ/ν; (Δv/sqrt(ν) - ( ( (ζ/sqrt(ν)) + Δ[1] )/( y[1]/sqrt(ν) + 1 ) )*(yv/ν))]
#     if norm(ρ[v_idx])>ρ[1]
#         return min(1.0, 1/(norm(ρ[v_idx]) - ρ[1]))
#     else
#         return 1.0
#     end
# end

@inline function linesearch(x::SVector{n,T},Δx::SVector{n,T},idx_ort::SVector{n_ort,Ti}, idx_soc::SVector{n_soc,Ti}) where {n,T,n_ort,n_soc,Ti}
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))

    x_ort  =  x[idx_ort]
    Δx_ort = Δx[idx_ort]
    x_soc  =  x[idx_soc]
    Δx_soc = Δx[idx_soc]

    min(ort_linesearch(x_ort,Δx_ort), soc_linesearch(x_soc, Δx_soc))
end
@inline function bring2cone(r::SVector{n,T},idx_ort::SVector{n_ort,Ti}, idx_soc::SVector{n_soc,Ti}) where {n,n_ort,n_soc,T,Ti}
    alpha = -1;

    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))
    r_ort = r[idx_ort]
    r_soc = r[idx_soc]

    if any(r_ort .<= 0)
        alpha = -minimum(r_ort);
    end

    res = r_soc[1] - norm(r_soc[SVector{n_soc-1}(2:n_soc)])
    if  (res <= 0)
        alpha = max(alpha,-res)
    end

    if alpha < 0
        return r
    else
        return r + (1 + alpha)*gen_e(idx_ort, idx_soc)
    end
end
@inline function initialize(c::SVector{nx,T},
                            G::SMatrix{ns,nx,T,nsnx},
                            h::SVector{ns,T},
                            idx_ort::SVector{n_ort,Ti},
                            idx_soc::SVector{n_soc,Ti}) where {nx,ns,nsnx,n_ort,n_soc,T,Ti}
    F = cholesky(Symmetric(G'*G))
    x̂ = F\(G'*h)
    s̃ = G*x̂ - h
    ŝ = bring2cone(s̃,idx_ort,idx_soc)

    x = F\(-c)
    z̃ = G*x

    ẑ = bring2cone(z̃,idx_ort,idx_soc)

    x̂,ŝ,ẑ
end

# TODO: cone product, inverse_cone_product, linesearch, and gen_e should use val
function solve_socp(c::SVector{nx,T},
                    G::SMatrix{ns,nx,T,nsnx},
                    h::SVector{ns,T},
                    idx_ort::SVector{n_ort,Ti},
                    idx_soc::SVector{n_soc,Ti};
                    pdip_tol::T=1e-4,
                    verbose::Bool = false) where {nx,ns,nsnx,n_ort,n_soc,T,Ti}

    x,s,z = initialize(c,G,h,idx_ort,idx_soc)

    if verbose
        @printf "iter     objv        gap       |Gx+s-h|      κ      step\n"
        @printf "---------------------------------------------------------\n"
    end

    e = gen_e(idx_ort, idx_soc)

    for main_iter = 1:20

        W = calc_NT_scalings(s,z,idx_ort,idx_soc)

        # cache normalized variables
        λ = W*z
        λλ = cone_product(λ,λ,idx_ort,idx_soc)

        # evaluate residuals
        rx = G'*z + c
        rz = s + G*x - h
        μ = dot(s,z)/(n_ort + 1)
        if μ < pdip_tol
            return x,s,z
        end

        # affine step
        bx = -rx
        λ_ds = inverse_cone_product(λ,-λλ,idx_ort, idx_soc)
        bz̃ = W\(-rz - W*(λ_ds))
        G̃ = W\G
        F = cholesky(Symmetric(G̃'*G̃))
        Δxa = F\(bx + G̃'*bz̃)
        Δza = W\(G̃*Δxa - bz̃)
        Δsa = W*(λ_ds - W*Δza)

        # linesearch on affine step
        αa = min(linesearch(s,Δsa,idx_ort, idx_soc), linesearch(z,Δza,idx_ort, idx_soc))
        ρ = dot(s + αa*Δsa, z + αa*Δza)/dot(s,z)
        σ = max(0, min(1,ρ))^3

        # centering and correcting step
        ds = -λλ - cone_product(W\Δsa, W*Δza,idx_ort, idx_soc) + σ*μ*e
        λ_ds = inverse_cone_product(λ,ds,idx_ort, idx_soc)
        bz̃ = W\(-rz - W*(λ_ds))
        Δx = F\(bx + G̃'*bz̃)
        Δz = W\(G̃*Δx - bz̃)
        Δs = W*(λ_ds - W*Δz)

        # final line search (.99 to avoid hitting edge of cone)
        α = min(1,0.99*min(linesearch(s,Δs,idx_ort, idx_soc), 0.99*linesearch(z,Δz,idx_ort, idx_soc)))

        # take step
        x += α*Δx
        s += α*Δs
        z += α*Δz

        if verbose
            @printf("%3d   %10.3e  %9.2e  %9.2e  %9.2e  % 6.4f\n",
              main_iter, c'*x, dot(s,z)/(n_ort + 1), norm(G*x + s - h),
              norm(cone_product(W\Δsa, W*Δza,idx_ort, idx_soc) + σ*μ*e), α)
        end



    end
    error("pdip failed")


end


# function tt()
#
#     c,G,h,idx_ort,idx_soc = build_pr()
#
#     x,s,z = solve_socp(c,G,h,idx_ort,idx_soc;verbose = true, pdip_tol = 1e-12)
#
#     @btime solve_socp($c,$G,$h,$idx_ort,$idx_soc; verbose = false)
#
# end
#
# tt()
