
struct NT_scaling_2{n_ort,n_soc1,n_soc1_sq,n_soc2,n_soc2_sq,T}
    ort::SVector{n_ort,T}
    soc1     ::SMatrix{n_soc1,n_soc1,T,n_soc1_sq}
    soc1_fact::Cholesky{T, SMatrix{n_soc1, n_soc1, T, n_soc1_sq}}
    soc2     ::SMatrix{n_soc2,n_soc2,T,n_soc2_sq}
    soc2_fact::Cholesky{T, SMatrix{n_soc2, n_soc2, T, n_soc2_sq}}
end
struct scaling_2{n_ort,n_soc1,n_soc1_sq,n_soc2,n_soc2_sq,T}
    ort::SVector{n_ort,T}
    soc1::SMatrix{n_soc1,n_soc1,T,n_soc1_sq}
    soc2::SMatrix{n_soc2,n_soc2,T,n_soc2_sq}
end

# solve linear systems with NT_scaling (W\g)
@inline function Base.:\(W1::NT_scaling_2{n_ort,n_soc1,n_soc1_sq,n_soc2,n_soc2_sq,T},g::SVector{r,T}) where {n_ort,n_soc1,n_soc1_sq,n_soc2,n_soc2_sq,T,r}
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc1 = SVector{n_soc1}((n_ort + 1):(n_ort + n_soc1))
    idx_soc2 = SVector{n_soc2}((n_ort + n_soc1 + 1):(n_ort + n_soc1 + n_soc2))
    [g[idx_ort] ./ W1.ort; W1.soc1_fact\g[idx_soc1]; W1.soc2_fact\g[idx_soc2]]
end

@inline function Base.:\(W1::NT_scaling_2{n_ort,n_soc1,n_soc1_sq,n_soc2,n_soc2_sq,T},W2::scaling_2{n_ort,n_soc1,n_soc1_sq,n_soc2,n_soc2_sq,T}) where {n_ort,n_soc1,n_soc1_sq,n_soc2,n_soc2_sq,T}
    ort = W2.ort ./ W1.ort
    soc1 = W1.soc1_fact\W2.soc1
    soc2 = W1.soc2_fact\W2.soc2
    scaling_2(ort,soc1,soc2)
end

# solve linear systems with a matrix on the rhs (W\G)
@inline @generated function Base.:\(W1::NT_scaling_2{n_ort,n_soc1,n_soc1_sq,n_soc2,n_soc2_sq,T},G::SMatrix{r,c,T,rc}) where {n_ort,n_soc1,n_soc1_sq,n_soc2,n_soc2_sq,T,r,c,rc}
    xi = [:(W1\G[:,$i]) for i = 1:c]
    quote
        hcat($(xi...))
    end
end

# matrix vector multiplication (W*b)
@inline function Base.:*(W1::NT_scaling_2{n_ort,n_soc1,n_soc1_sq,n_soc2,n_soc2_sq,T}, b::SVector{n,T}) where {n_ort,n_soc1,n_soc1_sq,n_soc2,n_soc2_sq,T,n}
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc1 = SVector{n_soc1}((n_ort + 1):(n_ort + n_soc1))
    idx_soc2 = SVector{n_soc2}((n_ort + n_soc1 + 1):(n_ort + n_soc1 + n_soc2))
    [W1.ort .* b[idx_ort]; W1.soc1*b[idx_soc1]; W1.soc2*b[idx_soc2]]
end
@inline @generated function Base.:*(W1::NT_scaling_2{n_ort,n_soc1,n_soc1_sq,n_soc2,n_soc2_sq,T}, G::SMatrix{r,c,T,rc}) where {n_ort,n_soc1,n_soc1_sq,n_soc2,n_soc2_sq,T,r,c,rc}
    xi = [:(W1*G[:,$i]) for i = 1:c]
    quote
        hcat($(xi...))
    end
end
@inline function Base.:*(W1::scaling_2{n_ort,n_soc1,n_soc1_sq,n_soc2,n_soc2_sq,T}, b::SVector{n,T}) where {n_ort,n_soc1,n_soc1_sq,n_soc2,n_soc2_sq,T,n}
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc1 = SVector{n_soc1}((n_ort + 1):(n_ort + n_soc1))
    idx_soc2 = SVector{n_soc2}((n_ort + n_soc1 + 1):(n_ort + n_soc1 + n_soc2))
    [W1.ort .* b[idx_ort]; W1.soc1*b[idx_soc1]; W1.soc2*b[idx_soc2]]
end
@inline @generated function Base.:*(W1::scaling_2{n_ort,n_soc1,n_soc1_sq,n_soc2,n_soc2_sq,T}, G::SMatrix{r,c,T,rc}) where {n_ort,n_soc1,n_soc1_sq,n_soc2,n_soc2_sq,T,r,c,rc}
    xi = [:(W1*G[:,$i]) for i = 1:c]
    quote
        hcat($(xi...))
    end
end
@inline function soc_NT_scaling(s_soc::SVector{n_soc,T}, z_soc::SVector{n_soc,T}) where {n_soc,T}
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
    W_soc1 = η*W̄
end
function calc_NT_scalings(s::SVector{n,T}, z::SVector{n,T}, idx_ort::SVector{n_ort,Ti}, idx_soc1::SVector{n_soc1,Ti}, idx_soc2::SVector{n_soc2,Ti}) where {n,T,n_ort,n_soc1,n_soc2,Ti}
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc1 = SVector{n_soc1}((n_ort + 1):(n_ort + n_soc1))
    idx_soc2 = SVector{n_soc2}((n_ort + n_soc1 + 1):(n_ort + n_soc1 + n_soc2))

    # ort
    W_ort = sqrt.(s[idx_ort] ./ z[idx_ort])

    # SOC
    W_soc1 = soc_NT_scaling(s[idx_soc1],z[idx_soc1])
    W_soc2 = soc_NT_scaling(s[idx_soc2],z[idx_soc2])

    # W_ort, W_soc
    NT_scaling_2(W_ort,W_soc1,cholesky(W_soc1), W_soc2, cholesky(W_soc2))
end
