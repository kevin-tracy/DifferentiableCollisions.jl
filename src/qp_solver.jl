
"""
Primal dual interior point method for solving the following problem
    min    0.5x'Qx + q'x
    st     Gx ≦ h

alg: https://stanford.edu/~boyd/papers/pdf/code_gen_impl.pdf

KKT systems are solved via reduction

reduction: Nocedal and Wright, Numerical Optimization, pg 482 (16.62)

This is for static arrays and is allocation free and fast.
"""

@inline function linesearch(x::SVector{nx,T}, dx::SVector{nx,T}) where {nx,T}
    # this returns the max α ∈ [0,1] st (x + Δx > 0)
    α = 1.0
    for i = 1:length(x)
        if dx[i]<0
            α = min(α,-x[i]/dx[i])
        end
    end
    return α
end

@inline function centering_params(s::SVector{ns,T},
                          z::SVector{ns,T},
                          s_a::SVector{ns,T},
                          z_a::SVector{ns,T}) where {ns,T}
    # mehrotra predictor-corrector
    μ = dot(s,z)/length(s)
    α = min(linesearch(s,s_a), linesearch(z,z_a))
    σ = (dot(s + α*s_a, z + α*z_a)/dot(s,z))^3
    return σ, μ
end

@inline function pdip_init(Q::SMatrix{nx,nx,T,nx_squared},
                   q::SVector{nx,T},
                   G::SMatrix{ns,nx,T,ns_nx},
                   h::SVector{ns,T}) where {nx,T,nx_squared,ns,ns_nx}
    # initialization for PDIP
    K = Symmetric(Q + G'*G)
    F = cholesky(K)
    x = F\(G'*h-q)
    z = G*x - h
    s = 1*z
    α_p = -minimum(-z)
    if α_p < 0
        s = -z
    else
        s = -z .+ (1 + α_p)
    end
    α_d = -minimum(z)
    if α_d >= 0
        z = z .+ (1 + α_d)
    end
    return x,s,z
end

@inline function solve_affine(Q::SMatrix{nx,nx,T,nx2},
                              G::SMatrix{nz,nx,T,nznx},
                              z::SVector{nz,T},
                              s::SVector{nz,T},
                              r1::SVector{nx,T},
                              r2::SVector{nz,T},
                              r3::SVector{nz,T}) where {nx,nz,nx2,nznx,T}

    # solve for affine step (nocedal and wright )
    invSZ = Diagonal(z ./ s)
    F = cholesky(Symmetric(Q + G'*invSZ*G))
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
function pdip(Q::SMatrix{nx,nx,T,nx_squared},
              q::SVector{nx,T},
              G::SMatrix{ns,nx,T,ns_nx},
              h::SVector{ns,T};
              verbose = false,
              tol = 1e-12) where {nx,T,nx_squared,ns,ns_nx}

    x,s,z = pdip_init(Q,q,G,h)
    for i = 1:25

        # evaluate residuals
        r1 = G'z + Q*x + q
        r2 = s .* z
        r3 = G*x + s - h

        # solve for affine step
        F, Δxa, Δsa, Δza = solve_affine(Q,G,z,s,r1,r2,r3)

        # corrector + centering step
        σ, μ = centering_params(s,z,Δsa,Δza)
        r2 = r2 - (σ*μ .- (Δsa .* Δza))
        Δx, Δs, Δz = solve_cc(F,G,z,s,r1,r2,r3)

        # line search
        α = min(1, 0.99*min(linesearch(s,Δs),linesearch(z,Δz)))
        x += α*Δx
        s += α*Δs
        z += α*Δz

        # termination criteria
        if verbose
            @show (dot(s,z)/length(s), α)
        end
        if dot(s,z)/length(s) < tol
            return x,s,z
        end
    end
    error("PDIP failed")
end
