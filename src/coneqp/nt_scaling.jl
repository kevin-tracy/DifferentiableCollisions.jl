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


@inline function ort_nt_scaling(s_ort::SVector{n,T}, z_ort::SVector{n,T}) where {n,T}
    Diagonal(sqrt.(s_ort ./ z_ort))
end
@inline function normalize_soc(x)
    xs = x[1]
    xv = x[SA[2,3,4]]
    x̄ = x*(1/sqrt(xs^2 - dot(xv,xv)))
end
@inline function soc_nt_scaling(s_soc::SVector{n,T}, z_soc::SVector{n,T}) where {n,T}
    J = Diagonal(SA[1,-1,-1,-1])
    z̄ = normalize_soc(z_soc)
    s̄ = normalize_soc(s_soc)
    γ = sqrt((1 + dot(z̄,s̄))/2)
    # w̄ = (1/(2*γ))*(s̄ + J*z̄)
    w̄ = (1/(2*γ))*(s̄ + [z̄[1];-z̄[SA[2,3,4]]])
    b = (1/(w̄[1] + 1))
    W̄_top = w̄'
    W̄_bot = hcat(w̄[SA[2,3,4]], (I + b*w̄[SA[2,3,4]]*w̄[SA[2,3,4]]'))
    W̄ = vcat(W̄_top, W̄_bot)
    W = W̄*((s_soc'*J*s_soc)/(z_soc'*J*z_soc))^(1/4)
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
    @show norm(W4 - SA_block_diag(W3.W_ort, W3.W_soc))

    @show W4

    @btime NT_matmul($W1,$W1)
    @btime $W2*$W2



end

ttt()
