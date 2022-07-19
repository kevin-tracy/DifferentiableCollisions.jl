cd("/Users/kevintracy/.julia/dev/DCD")
import Pkg; Pkg.activate(".")
using LinearAlgebra
using StaticArrays
using JLD2
using BenchmarkTools
using Printf
import DCD

function build_pr()
    @load "/Users/kevintracy/.julia/dev/DCD/extras/example_socp.jld2"

    nx = 5
    n_ort = length(h_ort)
    n_soc = length(h_soc)

    G = SMatrix{n_ort + n_soc, nx}([G_ort;G_soc])
    h = SVector{n_ort + n_soc}([h_ort;h_soc])

    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))

    c = SA[0,0,0,1,0.0]

    return c, G, h, idx_ort, idx_soc
end

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
    ν = max(y[1]^2 - dot(yv,yv), 1e-25) + 1e-14
    ζ = y[1]*Δ[1] - dot(yv,Δv)
    ρ = [ζ/ν; (Δv/sqrt(ν) - ( ( (ζ/sqrt(ν)) + Δ[1] )/( y[1]/sqrt(ν) + 1 ) )*(yv/ν))]
    if norm(ρ[v_idx])>ρ[1]
        return min(1.0, 1/(norm(ρ[v_idx]) - ρ[1]))
    else
        return 1.0
    end
end
@inline function linesearch(x::SVector{n,T},Δx::SVector{n,T},idx_ort::SVector{n_ort,Ti}, idx_soc::SVector{n_soc,Ti}) where {n,T,n_ort,n_soc,Ti}
    x_ort  =  x[idx_ort]
    Δx_ort = Δx[idx_ort]
    x_soc  =  x[idx_soc]
    Δx_soc = Δx[idx_soc]

    min(ort_linesearch(x_ort,Δx_ort), soc_linesearch(x_soc, Δx_soc))
end

function solve_socp(c::SVector{nx,T},
                    G::SMatrix{ns,nx,T,nsnx},
                    h::SVector{ns,T},
                    idx_ort::SVector{n_ort,Ti},
                    idx_soc::SVector{n_soc,Ti};
                    pdip_tol::T=1e-4,
                    verbose::Bool = true) where {nx,ns,nsnx,n_ort,n_soc,T,Ti}

    x = @SVector zeros(nx)
    s = [(@SVector ones(n_ort + 1)); .1*(@SVector ones(n_soc - 1))]
    z = [(@SVector ones(n_ort + 1)); .1*(@SVector ones(n_soc - 1))]

    if verbose
        @printf "iter     objv        gap       |Gx+s-h|      κ      step\n"
        @printf "---------------------------------------------------------\n"
    end

    for main_iter = 1:20

        # evaluate NT scaling
        W, W² = DCD.nt_scaling(s,z,idx_ort,idx_soc)
        λ = W*z
        λλ = DCD.cone_product(λ,λ,idx_ort,idx_soc)

        # evaluate residuals
        rx = G'*z + c
        rz = s + G*x - h
        μ = dot(s,z)/(n_ort + 1)
        if μ < pdip_tol
            # @info "Success"
            return x,s,z
        end

        # affine step
        bx = -rx
        λ_ds = DCD.inverse_cone_product(λ,-λλ,idx_ort, idx_soc)
        bz = -rz - W*(λ_ds)
        F = cholesky(Symmetric(G'*(W²\G)))
        Δxa = F\(bx + G'*(W²\bz))
        Δza = W²\(G*Δxa - bz)
        Δsa = W*(λ_ds - W*Δza)

        # linesearch on affine step
        αa = min(linesearch(s,Δsa,idx_ort, idx_soc), linesearch(z,Δza,idx_ort, idx_soc))
        ρ = dot(s + αa*Δsa, z + αa*Δza)/dot(s,z)
        σ = max(0, min(1,ρ))^3

        # centering and correcting step
        e = DCD.gen_e(idx_ort, idx_soc)
        ds = -λλ - DCD.cone_product(W\Δsa, W*Δza,idx_ort, idx_soc) + σ*μ*e
        λ_ds = DCD.inverse_cone_product(λ,ds,idx_ort, idx_soc)
        bz = -rz - W*(λ_ds)
        Δx = F\(bx + G'*(W²\bz))
        Δz = W²\(G*Δx - bz)
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
              norm(DCD.cone_product(W\Δsa, W*Δza,idx_ort, idx_soc) + σ*μ*e), α)
        end



    end
    error("pdip failed")


end


function tt()

    c,G,h,idx_ort,idx_soc = build_pr()

    # @show size(G)

    x,y,z = solve_socp(c,G,h,idx_ort,idx_soc)

    # @btime solve_socp($c,$G,$h,$idx_ort,$idx_soc)
    # x = @SVector randn(5)
    # s = [ones(θ.n_ort + 1); 0.01*ones(θ.n_soc-1) ]
    # z = copy(s) + 0.01*abs.(randn(length(s)))
    # x,s,z = init_coneqp(θ)

    # @printf "iter     objv        gap       |Gx+s-h|      κ      step\n"
    # @printf "---------------------------------------------------------\n"

    # c = θ.c
    # G = θ.G
    # h = θ.h
    # idx_x = θ.idx_x
    # idx_s = θ.idx_s
    # idx_z = θ.idx_z
    # idx_ort = θ.idx_ort
    # idx_soc = θ.idx_soc
    # m = length(idx_ort) + 1
    #
    # for main_iter = 1:30
    #
    #     # evaluate NT scaling
    #     W = nt_scaling(s,z,θ)
    #     λ = W*z
    #     λλ = cone_prod(λ,λ,θ)
    #
    #     # evaluate residuals
    #     rx = G'*z + c
    #     rz = s + G*x - h
    #     μ = dot(s,z)/m
    #     if μ < 1e-4
    #         @info "success"
    #         break
    #     end
    #
    #     # solve affine
    #     Δxa, Δsa, Δza =  solve_ls(-rx,-rz,-λλ,W,λ,θ)
    #
    #     αa = min(linesearch(s,Δsa,θ), linesearch(z,Δza,θ))
    #     ρ = dot(s + αa*Δsa, z + αa*Δza)/dot(s,z)
    #     σ = max(0, min(1,ρ))^3
    #
    #     η = 0.0
    #     γ = 1.0
    #     e = [ones(length(idx_ort)); gen_e(length(idx_soc))]
    #     ds = -λλ - γ*cone_prod((W')\Δsa, W*Δza,θ) + σ*μ*e
    #     Δx, Δs, Δz = solve_ls(-(1 - η)*rx,-(1 - η)*rz,ds,W,λ,θ)
    #
    #     α = min(1,0.99*min(linesearch(s,Δs,θ), 0.99*linesearch(z,Δz,θ)))
    #
    #     @printf("%3d   %10.3e  %9.2e  %9.2e  %9.2e  % 6.4f\n",
    #       main_iter, θ.c'*x, μ, norm(θ.G*x + s - θ.h),
    #       σ*μ, α)
    #     # @show α
    #     x += α*Δx
    #     s += α*Δs
    #     z += α*Δz
    # end

end

tt()
