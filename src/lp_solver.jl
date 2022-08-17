
"""
Primal dual interior point method for solving the following problem
    min    q'x
    st     Gx ≦ h

alg: https://stanford.edu/~boyd/papers/pdf/code_gen_impl.pdf

KKT systems are solved via reduction

reduction: Nocedal and Wright, Numerical Optimization, pg 482 (16.62)

This is for static arrays and is allocation free and fast.
"""

# @inline function linesearch(x::SVector{nx,T}, dx::SVector{nx,T}) where {nx,T}
#     # this returns the max α ∈ [0,1] st (x + Δx > 0)
#     α = 1.0
#     for i = 1:length(x)
#         if dx[i]<0
#             α = min(α,-x[i]/dx[i])
#         end
#     end
#     return α
# end

@inline function centering_params(s::SVector{ns,T},
                          z::SVector{ns,T},
                          s_a::SVector{ns,T},
                          z_a::SVector{ns,T}) where {ns,T}
    # mehrotra predictor-corrector
    μ = dot(s,z)/length(s)
    α = min(ort_linesearch(s,s_a), ort_linesearch(z,z_a))
    σ = (dot(s + α*s_a, z + α*z_a)/dot(s,z))^3
    return σ, μ
end

@inline function bring2cone(r::SVector{n,T}) where {n,T}
    alpha = -1

    if any(r .<= 0.0)
        alpha = -minimum(r)
    end

    if alpha < 0.0
        return r
    else
        return r .+ (1.0 + alpha)
    end
end

@inline function pdip_init(c::SVector{nx,T},
                           G::SMatrix{ns,nx,T,ns_nx},
                           h::SVector{ns,T}) where {nx,ns,ns_nx,T}
    # initialization for PDIP
    F = cholesky(Symmetric(G'*G))
    x̂ = F\(G'*h)
    s̃ = G*x̂ - h
    ŝ = bring2cone(s̃)

    x = F\(-c)
    z̃ = G*x

    ẑ = bring2cone(z̃)

    x̂,ŝ,ẑ
end

@inline function solve_affine(G::SMatrix{nz,nx,T,nznx},
                              z::SVector{nz,T},
                              s::SVector{nz,T},
                              r1::SVector{nx,T},
                              r2::SVector{nz,T},
                              r3::SVector{nz,T}) where {nx,nz,nznx,T}

    # solve for affine step (nocedal and wright )
    invSZ = Diagonal(z ./ s)
    F = cholesky(Symmetric(G'*invSZ*G))
    Δx = F\(-r1 + G'*invSZ*(-r3 + (r2 ./ z)))
    Δs = G*Δx + r3
    Δz = (r2 - (z .* Δs)) ./ s
    return F, Δx, -Δs, -Δz
end
@inline function solve_cc(F,G,z,s,r1,r2,r3)
    invSZ = Diagonal(z ./ s)
    Δx = F\(-r1 + G'*invSZ*(-r3 + (r2 ./ z)))
    Δs = G*Δx + r3
    Δz = (r2 - (z .* Δs)) ./ s
    return Δx, -Δs, -Δz
end
function solve_lp(q::SVector{nx,T},
                  G::SMatrix{ns,nx,T,ns_nx},
                  h::SVector{ns,T};
                  verbose = false,
                  pdip_tol = 1e-12) where {nx,T,ns,ns_nx}

    x,s,z = pdip_init(q,G,h)
    for i = 1:25

        # evaluate residuals
        r1 = G'z  + q
        r2 = s .* z
        r3 = G*x + s - h

        # solve for affine step
        F, Δxa, Δsa, Δza = solve_affine(G,z,s,r1,r2,r3)

        # corrector + centering step
        σ, μ = centering_params(s,z,Δsa,Δza)
        r2 = r2 - (σ*μ .- (Δsa .* Δza))
        Δx, Δs, Δz = solve_cc(F,G,z,s,r1,r2,r3)

        # line search
        α = min(1, 0.99*min(ort_linesearch(s,Δs),ort_linesearch(z,Δz)))
        x += α*Δx
        s += α*Δs
        z += α*Δz

        # termination criteria
        if verbose
            @show (norm(G*x + s - h), dot(s,z)/length(s), α)
        end
        if max(dot(s,z)/length(s),norm(r3)) < pdip_tol
            return x,s,z
        end
    end
    error("PDIP failed")
end
