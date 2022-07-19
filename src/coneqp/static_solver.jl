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
    ν = max(y[1]^2 - dot(yv,yv), 1e-25) + 1e-13
    ζ = y[1]*Δ[1] - dot(yv,Δv)
    ρ = [ζ/ν; (Δv/sqrt(ν) - ( ( (ζ/sqrt(ν)) + Δ[1] )/( y[1]/sqrt(ν) + 1 ) )*(yv/ν))]
    if norm(ρ[v_idx])>ρ[1]
        return min(1.0, 1/(norm(ρ[v_idx]) - ρ[1]))
    else
        return 1.0
    end
end
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
        return r + (1 + alpha)*DCD.gen_e(idx_ort, idx_soc)
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





function solve_socp(c::SVector{nx,T},
                    G::SMatrix{ns,nx,T,nsnx},
                    h::SVector{ns,T},
                    idx_ort::SVector{n_ort,Ti},
                    idx_soc::SVector{n_soc,Ti};
                    pdip_tol::T=1e-4,
                    verbose::Bool = false) where {nx,ns,nsnx,n_ort,n_soc,T,Ti}

    # x = @SVector ones(nx)
    # s = [(@SVector ones(n_ort + 1)); .1*(@SVector ones(n_soc - 1))]
    # z = [(@SVector ones(n_ort + 1)); .1*(@SVector ones(n_soc - 1))]
    x,s,z = initialize(c,G,h,idx_ort,idx_soc)

    if verbose
        @printf "iter     objv        gap       |Gx+s-h|      κ      step\n"
        @printf "---------------------------------------------------------\n"
    end

    e = DCD.gen_e(idx_ort, idx_soc)

    for main_iter = 1:20

        # evaluate NT scaling
        w_ort        = DCD.ort_nt_scaling_lite(s[idx_ort],z[idx_ort])
        w_soc, η_soc = DCD.soc_nt_scaling_lite(s[idx_soc],z[idx_soc])
        W = DCD.NT_lite(w_ort, w_soc, η_soc)

        # cache normalized variables
        λ = W*z
        λλ = DCD.cone_product(λ,λ,idx_ort,idx_soc)

        # evaluate residuals
        rx = G'*z + c
        rz = s + G*x - h
        μ = dot(s,z)/(n_ort + 1)
        if μ < pdip_tol
            return x,s,z
        end

        # affine step
        bx = -rx
        λ_ds = DCD.inverse_cone_product(λ,-λλ,idx_ort, idx_soc)
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
        ds = -λλ - DCD.cone_product(W\Δsa, W*Δza,idx_ort, idx_soc) + σ*μ*e
        λ_ds = DCD.inverse_cone_product(λ,ds,idx_ort, idx_soc)
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
              norm(DCD.cone_product(W\Δsa, W*Δza,idx_ort, idx_soc) + σ*μ*e), α)
        end



    end
    error("pdip failed")


end


function tt()

    c,G,h,idx_ort,idx_soc = build_pr()

    # @show size(G)

    x,s,z = solve_socp(c,G,h,idx_ort,idx_soc;verbose = true)
    # @show x
    # @show s
    # @show z

    @btime solve_socp($c,$G,$h,$idx_ort,$idx_soc; verbose = false)
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


# from JuMP
# value.(x) = [-2.397923630017413, -0.6291513676894855, 0.20605503080463847, 1.4203056811905659, -0.9231986608396786]
# dual.(c1) = [-1.9554403536818505e-9, -0.2438804212000268, -1.6925680654196474e-9, -2.6847818967476124e-9, -1.3124465760691362e-8, -1.0454388716239396, -7.97945869568145e-9, -1.2279337598158235e-8, -9.486007400989269e-9, -2.005727928458962e-8, -2.542072084308074e-9]
# dual.(c2) = [0.5739857780227104, 0.025243236422295225, -0.0641647911626764, 0.5698292087092]
#
# # from my solver
# x = [-2.4006291983958365, -0.6259193374250461, 0.20658738792595446, 1.4204354647371262, -0.9232544772695983]
# s = [1.8465393229538756, 3.036841467925623e-5, 2.138386721540207, 1.3477135955964736, 0.2737042086631979, 3.8080150533293094e-5, 0.45251076319271183, 0.29386888356623425, 0.38096388834972683, 0.17951522621899138, 1.4204372583422715, 0.8522630724568573, -0.04014124845094621, 0.09851765827722245, -0.8454527229369907]
# z = [1.424024776179708e-5, 0.24381211118536178, 2.0382479480547845e-6, 2.604515327170819e-5, 0.00011546380169133973, 1.045337364247085, 4.7570559062809535e-5, 3.99388486387077e-5, 5.2520566553717046e-5, 4.742724571084713e-5, 1.5398129737978408e-5, 0.573926996626629, 0.025187705334615958, -0.06416353097637557, 0.5697424327871496]
