# cd("/Users/kevintracy/.julia/dev/DCD")
# import Pkg; Pkg.activate(".")
# using LinearAlgebra
# using StaticArrays
# using JLD2
# using BenchmarkTools
# using Printf
# import DCD

# function build_pr()
#     @load "/Users/kevintracy/.julia/dev/DCD/extras/example_socp_2.jld2"
#
#     nx = 5
#     ns = n_ort + n_soc1 + n_soc2
#     idx_ort = SVector{n_ort}(1:n_ort)
#     idx_soc1 = SVector{n_soc1}((n_ort + 1):(n_ort + n_soc1))
#     idx_soc2 = SVector{n_soc2}((n_ort + n_soc1 + 1):(n_ort + n_soc1 + n_soc2))
#
#     c = SA[0,0,0,1,0.0]
#
#     return c, SMatrix{ns,nx}(G), SVector{ns}(h), idx_ort, idx_soc1, idx_soc2
# end

@inline function ort_linesearch(x::SVector{n,T},dx::SVector{n,T}) where {n,T}
    # this returns the max α ∈ [0,1] st (x + Δx > 0)
    α = 1.0
    for i = 1:length(x)
        if dx[i]<0
            α = min(α,-x[i]/dx[i])
        end
    end
    return α
end

@inline function soc_linesearch(y::SVector{n,T},Δ::SVector{n,T}) where {n,T}
    v_idx = SVector{n-1}(2:n)
    yv = y[v_idx]
    Δv = Δ[v_idx]
    ν = max(y[1]^2 - dot(yv,yv), 1e-25) + 1e-13
    ζ = y[1]*Δ[1] - dot(yv,Δv)
    ρ = [ζ/ν; (Δv/sqrt(ν) - ( ( (ζ/sqrt(ν)) + Δ[1] )/( y[1]/sqrt(ν) + 1 ) )*(yv/ν))]
    if norm(ρ[v_idx])>ρ[1]
        return min(1.0, 1/(norm(ρ[v_idx]) - ρ[1]))
    else
        return 1.0
    end
end
@inline function linesearch(x::SVector{n,T},Δx::SVector{n,T},idx_ort::SVector{n_ort,Ti}, idx_soc1::SVector{n_soc1,Ti},idx_soc2::SVector{n_soc2,Ti}) where {n,T,n_ort,n_soc1, n_soc2,Ti}
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc1 = SVector{n_soc1}((n_ort + 1):(n_ort + n_soc1))
    idx_soc2 = SVector{n_soc2}((n_ort + n_soc1 + 1):(n_ort + n_soc1 + n_soc2))

    x_ort  =  x[idx_ort]
    Δx_ort = Δx[idx_ort]
    x_soc1  =  x[idx_soc1]
    Δx_soc1 = Δx[idx_soc1]
    x_soc2  =  x[idx_soc2]
    Δx_soc2 = Δx[idx_soc2]

    min(ort_linesearch(x_ort,Δx_ort), soc_linesearch(x_soc1, Δx_soc1),soc_linesearch(x_soc2, Δx_soc2))
end
@inline function bring2cone(r::SVector{n,T},idx_ort::SVector{n_ort,Ti}, idx_soc1::SVector{n_soc1,Ti},idx_soc2::SVector{n_soc2,Ti}) where {n,n_ort,n_soc1,n_soc2,T,Ti}
    alpha = -1;

    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc1 = SVector{n_soc1}((n_ort + 1):(n_ort + n_soc1))
    idx_soc2 = SVector{n_soc2}((n_ort + n_soc1 + 1):(n_ort + n_soc1 + n_soc2))
    r_ort = r[idx_ort]
    r_soc1 = r[idx_soc1]
    r_soc2 = r[idx_soc2]

    if any(r_ort .<= 0)
        alpha = -minimum(r_ort);
    end

    # SOC's
    res = r_soc1[1] - norm(r_soc1[SVector{n_soc1-1}(2:n_soc1)])
    if  (res <= 0)
        alpha = max(alpha,-res)
    end
    res = r_soc2[1] - norm(r_soc2[SVector{n_soc2-1}(2:n_soc2)])
    if  (res <= 0)
        alpha = max(alpha,-res)
    end

    if alpha < 0
        return r
    else
        return r + (1 + alpha)*gen_e(idx_ort, idx_soc1,idx_soc2)
    end
end
@inline function initialize(c::SVector{nx,T},
                            G::SMatrix{ns,nx,T,nsnx},
                            h::SVector{ns,T},
                            idx_ort::SVector{n_ort,Ti},
                            idx_soc1::SVector{n_soc1,Ti},
                            idx_soc2::SVector{n_soc2,Ti}) where {nx,ns,nsnx,n_ort,n_soc1,n_soc2,T,Ti}
    F = cholesky(Symmetric(G'*G))
    x̂ = F\(G'*h)
    s̃ = G*x̂ - h
    ŝ = bring2cone(s̃,idx_ort,idx_soc1,idx_soc2)

    x = F\(-c)
    z̃ = G*x

    ẑ = bring2cone(z̃,idx_ort,idx_soc1,idx_soc2)

    x̂,ŝ,ẑ
end

# TODO: cone product, inverse_cone_product, linesearch, and gen_e should use val
function solve_socp(c::SVector{nx,T},
                    G::SMatrix{ns,nx,T,nsnx},
                    h::SVector{ns,T},
                    idx_ort::SVector{n_ort,Ti},
                    idx_soc1::SVector{n_soc1,Ti},
                    idx_soc2::SVector{n_soc2,Ti};
                    pdip_tol::T=1e-4,
                    verbose::Bool = false) where {nx,ns,nsnx,n_ort,n_soc1,n_soc2,T,Ti}

    x,s,z = initialize(c,G,h,idx_ort,idx_soc1,idx_soc2)

    if verbose
        @printf "iter     objv        gap       |Gx+s-h|      κ      step\n"
        @printf "---------------------------------------------------------\n"
    end

    e = gen_e(idx_ort, idx_soc1,idx_soc2)

    for main_iter = 1:20

        W = calc_NT_scalings(s,z,idx_ort,idx_soc1,idx_soc2)

        # cache normalized variables
        λ = W*z
        λλ = cone_product(λ,λ,idx_ort,idx_soc1,idx_soc2)

        # evaluate residuals
        rx = G'*z + c
        rz = s + G*x - h
        μ = dot(s,z)/(n_ort + 1)
        if μ < pdip_tol
            return x,s,z
        end

        # affine step
        bx = -rx
        λ_ds = inverse_cone_product(λ,-λλ,idx_ort,idx_soc1,idx_soc2)
        bz̃ = W\(-rz - W*(λ_ds))
        G̃ = W\G
        F = cholesky(Symmetric(G̃'*G̃))
        Δxa = F\(bx + G̃'*bz̃)
        Δza = W\(G̃*Δxa - bz̃)
        Δsa = W*(λ_ds - W*Δza)

        # linesearch on affine step
        αa = min(linesearch(s,Δsa,idx_ort, idx_soc1,idx_soc2), linesearch(z,Δza,idx_ort,idx_soc1,idx_soc2))
        ρ = dot(s + αa*Δsa, z + αa*Δza)/dot(s,z)
        σ = max(0, min(1,ρ))^3

        # centering and correcting step
        ds = -λλ - cone_product(W\Δsa, W*Δza,idx_ort,idx_soc1,idx_soc2) + σ*μ*e
        λ_ds = inverse_cone_product(λ,ds,idx_ort,idx_soc1,idx_soc2)
        bz̃ = W\(-rz - W*(λ_ds))
        Δx = F\(bx + G̃'*bz̃)
        Δz = W\(G̃*Δx - bz̃)
        Δs = W*(λ_ds - W*Δz)

        # final line search (.99 to avoid hitting edge of cone)
        α = min(1,0.99*min(linesearch(s,Δs,idx_ort,idx_soc1,idx_soc2), 0.99*linesearch(z,Δz,idx_ort,idx_soc1,idx_soc2)))

        # take step
        x += α*Δx
        s += α*Δs
        z += α*Δz

        if verbose
            @printf("%3d   %10.3e  %9.2e  %9.2e  %9.2e  % 6.4f\n",
              main_iter, c'*x, dot(s,z)/(n_ort + 1), norm(G*x + s - h),
              norm(cone_product(W\Δsa, W*Δza,idx_ort,idx_soc1,idx_soc2) + σ*μ*e), α)
        end



    end
    error("pdip failed")


end


# function tt()
#
#     c,G,h,idx_ort,idx_soc1,idx_soc2 = build_pr()
#
#     x,s,z = solve_socp(c,G,h,idx_ort,idx_soc1,idx_soc2;verbose = true, pdip_tol = 1e-12)
#
#     @btime solve_socp($c,$G,$h,$idx_ort,$idx_soc1,$idx_soc2; verbose = false)
#
# end
#
# tt()
