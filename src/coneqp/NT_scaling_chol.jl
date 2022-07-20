struct NT_scaling{n_ort,n_soc,n_soc2,T}
    ort::SVector{n_ort,T}
    soc::SMatrix{n_soc,n_soc,T,n_soc2}
    soc_fact::Cholesky{T, SMatrix{n_soc, n_soc, T, n_soc2}}
end

@inline function normalize_soc(x::SVector{nx,T}) where {nx, T}
    x/sqrt(soc_quad_J(x))
end
@inline function soc_quad_J(x::SVector{nx,T}) where {nx,T}
    xs = x[1]
    xv = x[SVector{nx-1}(2:nx)]
    xs^2 - dot(xv,xv)
end

@inline function Base.:\(W1::NT_scaling{n_ort,n_soc,n_soc2,T},g::SVector{r,T}) where {n_ort,n_soc,n_soc2,T,r}
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))
    [g[idx_ort] ./ W1.ort; W1.soc_fact\g[idx_soc]]
end
# NOTE: this is slower than the @generated version
# @inline function Base.:\(W1::NT2{n_ort,n_soc,n_soc2,T},G::SMatrix{r,c,T,rc}) where {n_ort,n_soc,n_soc2,T,r,c,rc}
#     idx_ort = SVector{n_ort}(1:n_ort)
#     idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))
#     [W1.W_ort\G[idx_ort,:]; W1.W_soc_fact\G[idx_soc,:]]
# end
@inline @generated function Base.:\(W1::NT_scaling{n_ort,n_soc,n_soc2,T},G::SMatrix{r,c,T,rc}) where {n_ort,n_soc,n_soc2,T,r,c,rc}
    xi = [:(W1\G[:,$i]) for i = 1:c]
    quote
        hcat($(xi...))
    end
end
@inline function Base.:*(W1::NT_scaling{n_ort,n_soc,n_soc2,T}, b::SVector{n,T}) where {n_ort,n_soc,n,T,n_soc2}
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))
    [W1.ort .* b[idx_ort]; W1.soc*b[idx_soc]]
end

function calc_NT_scalings(s::SVector{n,T}, z::SVector{n,T}, idx_ort::SVector{n_ort,Ti}, idx_soc::SVector{n_soc,Ti}) where {n,T,n_ort,n_soc,Ti}
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))

    # ort
    W_ort = sqrt.(s[idx_ort] ./ z[idx_ort])

    # SOC
    s_soc = s[idx_soc]
    z_soc = z[idx_soc]
    v_idx = SVector{n_soc-1}(2:n_soc)
    z̄ = normalize_soc(z_soc)
    s̄ = normalize_soc(s_soc)
    γ = sqrt((1 + dot(z̄,s̄))/2)
    w̄ = (1/(2*γ))*(s̄ + [z̄[1];-z̄[v_idx]])
    b = (1/(w̄[1] + 1))
    W̄_top = w̄'
    W̄_bot = hcat(w̄[v_idx], (I + b*(w̄[v_idx]*w̄[v_idx]')))
    W̄ = vcat(W̄_top, W̄_bot)
    η = (soc_quad_J(s_soc)/soc_quad_J(z_soc))^(1/4)
    W_soc = η*W̄

    # W_ort, W_soc
    NT_scaling(W_ort,W_soc,cholesky(W_soc))
end

function SA_block_diag(W1::Diagonal{T, SVector{n, T}}, W2::SMatrix{m,m,T,m2}) where {n,m,m2,T}
    # NOTE: this is only for testing
    top = hcat(W1, (@SMatrix zeros(n,m)))
    bot = hcat((@SMatrix zeros(m,n)), W2)
    vcat(top,bot)
end
