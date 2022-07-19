using LinearAlgebra
using StaticArrays
using BenchmarkTools

struct NT{n_ort,n_soc,n_soc2,T}
    W_ort::Diagonal{T, SVector{n_ort, T}}
    W_soc::SMatrix{n_soc,n_soc,T,n_soc2}
    W_ort_inv::Diagonal{T, SVector{n_ort, T}}
    W_soc_inv::SMatrix{n_soc,n_soc,T,n_soc2}
end

struct NT_lite{n_ort,n_soc,T}
    w_ort::SVector{n_ort,T}
    w_soc::SVector{n_soc,T}
    η_soc::T
end
# @inline function square_NT!(W::NT{n,m,e,T}) where {n,m,e,T}
#     # NT{n,m,e,T}(W.W_ort^2,W.W_soc^2,W.W_ort_inv^2, W.W_soc_inv^2)
#     W.W_ort = W.W_ort^2
#     W.W_soc = W.W_soc^2
#     W.W_ort_inv = W.W_ort_inv^2
#     W.W_soc_inv = W.W_soc_inv^2
#     return nothing
# end

# now let's do W * matrix
@inline function NT_mat_mul(W1::NT{n,m,e,T},G::SMatrix{r,c,T,rc}) where {n,m,e,T,r,c,rc}
    idx_ort = SVector{n}(1:n)
    idx_soc = SVector{m}((n + 1):(n + m))

    vcat(W1.W_ort*G[idx_ort,:], W1.W_soc*G[idx_soc,:])
end

@inline function NT_vec_mul(W1::NT{n,m,e,T},G::SVector{r,T}) where {n,m,e,T,r}
    idx_ort = SVector{n}(1:n)
    idx_soc = SVector{m}((n + 1):(n + m))

    vcat(W1.W_ort*G[idx_ort], W1.W_soc*G[idx_soc])
end


@inline function NT_lin_solve(W1::NT{n,m,e,T},G::SVector{r,T}) where {n,m,e,T,r}
    idx_ort = SVector{n}(1:n)
    idx_soc = SVector{m}((n + 1):(n + m))

    vcat(W1.W_ort_inv*G[idx_ort], W1.W_soc_inv*G[idx_soc])
end

@inline function NT_lin_solve_mat(W1::NT{n,m,e,T},G::SMatrix{r,c,T,rc}) where {n,m,e,T,r,c,rc}
    idx_ort = SVector{n}(1:n)
    idx_soc = SVector{m}((n + 1):(n + m))

    vcat(W1.W_ort_inv*G[idx_ort,:], W1.W_soc_inv*G[idx_soc,:])
end

# import Base: *
# @inline (*)(W1::NT{n,m,e,T},W2::NT{n,m,e,T}) where {n,m,e,T} = NT_matmul(W1::NT{n,m,e,T},W2::NT{n,m,e,T})
# @inline (*)(W1::NT{n,m,e,T},G::SMatrix{r,c,T,rc}) where {n,m,e,T,r,c,rc} = NT_matmul(W1::NT{n,m,e,T},G::SMatrix{r,c,T,rc})

# NOTE: deprecated
# @inline function Base.:*(W1::NT{n,m,e,T},W2::NT{n,m,e,T}) where {n,m,e,T}
#     NT_mul(W1::NT{n,m,e,T},W2::NT{n,m,e,T})
# end

@inline function Base.:*(W1::NT{n,m,e,T},G::SMatrix{r,c,T,rc}) where {n,m,e,T,r,c,rc}
    NT_mat_mul(W1::NT{n,m,e,T},G::SMatrix{r,c,T,rc})
end
@inline function Base.:*(W1::NT{n,m,e,T},G::SVector{r,T}) where {n,m,e,T,r}
    NT_vec_mul(W1::NT{n,m,e,T},G::SVector{r,T})
end
@inline function Base.:*(W::NT_lite{n_ort,n_soc,T}, z::SVector{n,T}) where {n_ort,n_soc,n,T}
    NT_vec_mul_lite(W::NT_lite{n_ort,n_soc,T}, z::SVector{n,T})
end
@inline function Base.:\(W1::NT{n,m,e,T},G::SVector{r,T}) where {n,m,e,T,r}
    NT_lin_solve(W1::NT{n,m,e,T},G::SVector{r,T})
end
@inline function Base.:\(W::NT_lite{n_ort,n_soc,T}, z::SVector{n,T}) where {n_ort,n_soc,n,T}
    NT_lin_solve_lite(W::NT_lite{n_ort,n_soc,T}, z::SVector{n,T})
end
@inline function Base.:\(W1::NT{n,m,e,T},G::SMatrix{r,c,T,rc}) where {n,m,e,T,r,c,rc}
    NT_lin_solve_mat(W1::NT{n,m,e,T},G::SMatrix{r,c,T,rc})
end
@inline function Base.:\(W::NT_lite{n_ort,n_soc,T}, Z::SMatrix{r,c,T,rc}) where {n_ort,n_soc,r,c,rc,T}
    NT_lin_solve_mat_lite(W::NT_lite{n_ort,n_soc,T}, Z::SMatrix{r,c,T,rc})
end

@inline function ort_nt_scaling(s_ort::SVector{n,T}, z_ort::SVector{n,T}) where {n,T}
    v = sqrt.(s_ort ./ z_ort)
    W = Diagonal(v)
    Winv = Diagonal( 1 ./ v)
    return W, Winv
end
@inline function normalize_soc(x::SVector{nx,T}) where {nx, T}
    x/sqrt(soc_quad_J(x))
end
@inline function soc_quad_J(x::SVector{nx,T}) where {nx,T}
    xs = x[1]
    xv = x[SVector{nx-1}(2:nx)]
    xs^2 - dot(xv,xv)
end
@inline function ort_nt_scaling_lite(s_ort::SVector{n,T}, z_ort::SVector{n,T}) where {n,T}
    sqrt.(s_ort ./ z_ort)
end
@inline function soc_nt_scaling_lite(s_soc::SVector{n,T}, z_soc::SVector{n,T}) where {n,T}
    v_idx = SVector{n-1}(2:n)
    sres = soc_quad_J(s_soc)
    zres = soc_quad_J(z_soc)
    s̄ = s_soc/sqrt(sres)
    z̄ = z_soc/sqrt(zres)
    γ = sqrt((1 + dot(z̄,s̄))/2)
    w̄ = (1/(2*γ))*(s̄ + [z̄[1];-z̄[v_idx]])
    η = (sres/zres)^(1/4)
    w̄, η
end
@inline function NT_vec_mul_lite(W::NT_lite{n_ort,n_soc,T}, z::SVector{n,T}) where {n_ort,n_soc,n,T}
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))

    w̄ = W.w_soc
    v_idx = SVector{n_soc-1}(2:n_soc)
    w̄0 = w̄[1]
    w̄1 = w̄[v_idx]
    z_soc = z[idx_soc]
    z0 = z_soc[1]
    z1 = z_soc[v_idx]
    ζ = dot(w̄1,z1)
    Wz = W.η_soc*[w̄0*z0 + ζ; z1 + (z0 + ζ/(1 + w̄0))*w̄1]

    [W.w_ort .* z[idx_ort]; Wz]
end
@inline function NT_lin_solve_lite(W::NT_lite{n_ort,n_soc,T}, z::SVector{n,T})::SVector{n,T} where {n_ort,n_soc,n,T}
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))

    w̄ = W.w_soc
    v_idx = SVector{n_soc-1}(2:n_soc)
    w̄0 = w̄[1]
    w̄1 = w̄[v_idx]
    z_soc = z[idx_soc]
    z0 = z_soc[1]
    z1 = z_soc[v_idx]
    ζ = dot(w̄1,z1)
    Winvz = (1/W.η_soc)*[w̄0*z0 - ζ; z1 + (-z0 + ζ/(1 + w̄0))*w̄1]

    [z[idx_ort] ./ W.w_ort ; Winvz]
end
@inline function squared_solve(W::NT_lite{n_ort,n_soc,T}, z::SVector{n,T})::SVector{n,T} where {n_ort,n_soc,n,T}
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))

    w̄ = W.w_soc
    v_idx = SVector{n_soc-1}(2:n_soc)
    w̄0 = w̄[1]
    w̄1 = w̄[v_idx]

    # first solve
    z_soc = z[idx_soc]
    z0 = z_soc[1]
    z1 = z_soc[v_idx]
    ζ = dot(w̄1,z1)
    Winvz = (1/W.η_soc)*[w̄0*z0 - ζ; z1 + (-z0 + ζ/(1 + w̄0))*w̄1]

    # second solve
    z0 = Winvz[1]
    z1 = Winvz[v_idx]
    ζ = dot(w̄1,z1)
    W²invz = (1/W.η_soc)*[w̄0*z0 - ζ; z1 + (-z0 + ζ/(1 + w̄0))*w̄1]

    [z[idx_ort] ./ (W.w_ort).^2 ; W²invz]
end
@inline @generated function NT_lin_solve_mat_lite(W::NT_lite{n_ort,n_soc,T}, Z::SMatrix{r,c,T,rc}) where {n_ort,n_soc,r,c,rc,T}

    # here we generate the following:
    # vcat(
    # NT_lin_solve_lite(W, Z[:,1]),
    # NT_lin_solve_lite(W, Z[:,2]),
    # NT_lin_solve_lite(W, Z[:,3]),
    # ... # continued up to c
    # )
    xi = [:(NT_lin_solve_lite(W, Z[:,$i])) for i = 1:c]
    quote
        hcat($(xi...))
    end
end
@inline function soc_nt_scaling(s_soc::SVector{n,T}, z_soc::SVector{n,T}) where {n,T}
    # J = Diagonal(SA[1,-1,-1,-1])
    v_idx = SVector{n-1}(2:n)
    z̄ = normalize_soc(z_soc)
    s̄ = normalize_soc(s_soc)
    γ = sqrt((1 + dot(z̄,s̄))/2)
    # w̄ = (1/(2*γ))*(s̄ + J*z̄)
    w̄ = (1/(2*γ))*(s̄ + [z̄[1];-z̄[v_idx]])
    b = (1/(w̄[1] + 1))
    W̄_top = w̄'
    W̄_bot = hcat(w̄[v_idx], (I + b*(w̄[v_idx]*w̄[v_idx]')))
    W̄ = vcat(W̄_top, W̄_bot)
    η = (soc_quad_J(s_soc)/soc_quad_J(z_soc))^(1/4)
    W = η*W̄

    W̄inv_top = hcat(w̄[1], -w̄[v_idx]')
    W̄inv_bot = hcat(-w̄[v_idx], (I + b*(w̄[v_idx]*w̄[v_idx]')))
    W̄inv = vcat(W̄inv_top, W̄inv_bot)
    Winv = (1/η)*W̄inv
    return W, Winv
end
function SA_block_diag(W1::Diagonal{T, SVector{n, T}}, W2::SMatrix{m,m,T,m2}) where {n,m,m2,T}
    top = hcat(W1, (@SMatrix zeros(n,m)))
    bot = hcat((@SMatrix zeros(m,n)), W2)
    vcat(top,bot)
end

function nt_scaling(s::SVector{ns,T},z::SVector{ns,T},idx_ort::SVector{n_ort,Ti},idx_soc::SVector{n_soc,Ti}) where {ns,T,n_ort,n_soc,Ti}
    W_ort, W_ort_inv = DCD.ort_nt_scaling(s[idx_ort], z[idx_ort])
    W_soc, W_soc_inv = DCD.soc_nt_scaling(s[idx_soc], z[idx_soc])
    W = DCD.NT(W_ort, W_soc, W_ort_inv, W_soc_inv)
    W² = DCD.NT(W_ort^2, W_soc^2, W_ort_inv^2, W_soc_inv^2)
    return W, W²
end

# function ttt()
#
#     n_ort = 8
#     n_soc = 4
#     idx_ort = SVector{n_ort}(1:n_ort)
#     idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))
#
#     ns = n_ort + n_soc
#     s = SVector{ns}([abs.(randn(n_ort)); 10; abs.(randn(n_soc - 1))])
#     z = SVector{ns}([abs.(randn(n_ort)); 10; abs.(randn(n_soc - 1))])
#
#     W_ort = ort_nt_scaling(s[idx_ort], z[idx_ort])
#     W_soc = soc_nt_scaling(s[idx_soc], z[idx_soc])
#
#     W1 = NT(W_ort, W_soc)
#     W2 = SA_block_diag(W_ort,W_soc)
#
#     # W3 = NT_matmul(W1,W1)
#     W3 = W1*W1
#     W4 = W2*W2
#     @assert norm(W4 - SA_block_diag(W3.W_ort, W3.W_soc))<1e-13
#
#     # @show W4
#
#     # @btime NT_matmul($W1,$W1)
#     # @btime $W2*$W2
#     #
#     # x = SA[10,1,2,3,2]
#     #
#     # x2 = normalize_soc(x)
#     #
#     # @btime normalize_soc($x)
#
#     G = @SMatrix randn(n_soc + n_ort, 4)
#
#     O = W2*G
#     O2 = [W_ort*G[idx_ort,:]; W_soc*G[idx_soc,:]]
#     # O3 = NT_matmul(W1,G)
#     O3 = W1*G
#
#     @show norm(O - O2)
#     @show norm(O - O3)
#
#     # @btime NT_matmul($W1,$G)
#     @btime $W1*$G
#     @btime $W2*$G
#
#
# end

# ttt()
