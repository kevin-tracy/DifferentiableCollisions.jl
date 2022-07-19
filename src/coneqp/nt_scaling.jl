
struct NT{n_ort,n_soc,n_soc2,T}
    W_ort::Diagonal{T, SVector{n_ort, T}}
    W_soc::SMatrix{n_soc,n_soc,T,n_soc2}
    W_ort_inv::Diagonal{T, SVector{n_ort, T}}
    W_soc_inv::SMatrix{n_soc,n_soc,T,n_soc2}
end

# @inline function NT_mat_mul(W1::NT{n,m,e,T},G::SMatrix{r,c,T,rc}) where {n,m,e,T,r,c,rc}
#     idx_ort = SVector{n}(1:n)
#     idx_soc = SVector{m}((n + 1):(n + m))
#
#     vcat(W1.W_ort*G[idx_ort,:], W1.W_soc*G[idx_soc,:])
# end

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


# @inline function Base.:*(W1::NT{n,m,e,T},G::SMatrix{r,c,T,rc}) where {n,m,e,T,r,c,rc}
#     NT_mat_mul(W1::NT{n,m,e,T},G::SMatrix{r,c,T,rc})
# end
@inline function Base.:*(W1::NT{n,m,e,T},G::SVector{r,T}) where {n,m,e,T,r}
    NT_vec_mul(W1::NT{n,m,e,T},G::SVector{r,T})
end
@inline function Base.:\(W1::NT{n,m,e,T},G::SVector{r,T}) where {n,m,e,T,r}
    NT_lin_solve(W1::NT{n,m,e,T},G::SVector{r,T})
end
@inline function Base.:\(W1::NT{n,m,e,T},G::SMatrix{r,c,T,rc}) where {n,m,e,T,r,c,rc}
    NT_lin_solve_mat(W1::NT{n,m,e,T},G::SMatrix{r,c,T,rc})
end


@inline function ort_nt_scaling(s_ort::SVector{n,T}, z_ort::SVector{n,T}) where {n,T}
    v = sqrt.(s_ort ./ z_ort)
    W = Diagonal(v)
    Winv = Diagonal( 1 ./ v)
    return W, Winv
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


function nt_scaling(s::SVector{ns,T},z::SVector{ns,T},idx_ort::SVector{n_ort,Ti},idx_soc::SVector{n_soc,Ti}) where {ns,T,n_ort,n_soc,Ti}
    W_ort, W_ort_inv = DCD.ort_nt_scaling(s[idx_ort], z[idx_ort])
    W_soc, W_soc_inv = DCD.soc_nt_scaling(s[idx_soc], z[idx_soc])
    W = DCD.NT(W_ort, W_soc, W_ort_inv, W_soc_inv)
    W² = DCD.NT(W_ort^2, W_soc^2, W_ort_inv^2, W_soc_inv^2)
    return W, W²
end
