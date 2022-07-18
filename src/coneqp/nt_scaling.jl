using LinearAlgebra
using StaticArrays
using BenchmarkTools

struct NT{n_ort,n_soc,n_soc2,T}
    W_ort::Diagonal{T, SVector{n_ort, T}}
    W_soc::SMatrix{n_soc,n_soc,T,n_soc2}
end

@inline function NT_matmul(W1::NT{n,m,e,T},W2::NT{n,m,e,T}) where {n,m,e,T}
    W_ort = W1.W_ort*W2.W_ort
    W_soc = W1.W_soc*W2.W_soc
    NT{n,m,e,T}(W_ort,W_soc)
end

import Base: *
@inline (*)(W1::NT{n,m,e,T},W2::NT{n,m,e,T}) where {n,m,e,T} = NT_matmul(W1::NT{n,m,e,T},W2::NT{n,m,e,T})

# now let's do W * matrix
@inline function NT_matmul(W1::NT{n,m,e,T},G::SMatrix{r,c,T,rc}) where {n,m,e,T,r,c,rc}
    idx_ort = SVector{n}(1:n)
    idx_soc = SVector{m}((n + 1):(n + m))
    # W_ort = W1.W_ort*W2.W_ort
    # W_soc = W1.W_soc*W2.W_soc
    # NT{n,m,e,T}(W_ort,W_soc)
end

@inline function ort_nt_scaling(s_ort::SVector{n,T}, z_ort::SVector{n,T}) where {n,T}
    Diagonal(sqrt.(s_ort ./ z_ort))
end
@inline function normalize_soc(x::SVector{nx,T}) where {nx, T}
    # xs = x[1]
    # xv = x[SVector{nx-1}(2:nx)]
    # x̄ = x*(1/sqrt(xs^2 - dot(xv,xv)))
    x/sqrt(soc_quad_J(x))
end
@inline function soc_quad_J(x::SVector{nx,T}) where {nx,T}
    xs = x[1]
    xv = x[SVector{nx-1}(2:nx)]
    xs^2 - dot(xv,xv)
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
    W = W̄*((soc_quad_J(s_soc)/soc_quad_J(z_soc))^(1/4))
end
function SA_block_diag(W1::Diagonal{T, SVector{n, T}}, W2::SMatrix{m,m,T,m2}) where {n,m,m2,T}
    top = hcat(W1, (@SMatrix zeros(n,m)))
    bot = hcat((@SMatrix zeros(m,n)), W2)
    vcat(top,bot)
end


function ttt()

    n_ort = 8
    n_soc = 4
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))

    ns = n_ort + n_soc
    s = SVector{ns}([abs.(randn(n_ort)); 10; abs.(randn(n_soc - 1))])
    z = SVector{ns}([abs.(randn(n_ort)); 10; abs.(randn(n_soc - 1))])

    W_ort = ort_nt_scaling(s[idx_ort], z[idx_ort])
    W_soc = soc_nt_scaling(s[idx_soc], z[idx_soc])

    W1 = NT(W_ort, W_soc)
    W2 = SA_block_diag(W_ort,W_soc)

    # W3 = NT_matmul(W1,W1)
    W3 = W1*W1
    W4 = W2*W2
    @assert norm(W4 - SA_block_diag(W3.W_ort, W3.W_soc))<1e-13

    # @show W4

    # @btime NT_matmul($W1,$W1)
    # @btime $W2*$W2
    #
    # x = SA[10,1,2,3,2]
    #
    # x2 = normalize_soc(x)
    #
    # @btime normalize_soc($x)

    G = @SMatrix randn(n_soc + n_ort, 4)

    O = W2*G
    O2 = [W_ort*G[idx_ort,:]; W_soc*G[idx_soc,:]]

    @show norm(O - O2)

    @btime NT_matmul($W1,$G)


end

ttt()
