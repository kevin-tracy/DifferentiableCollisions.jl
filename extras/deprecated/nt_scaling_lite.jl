
struct NT_lite{n_ort,n_soc,T}
    w_ort::SVector{n_ort,T}
    w_soc::SVector{n_soc,T}
    η_soc::T
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
# @inline function squared_solve(W::NT_lite{n_ort,n_soc,T}, z::SVector{n,T})::SVector{n,T} where {n_ort,n_soc,n,T}
#     idx_ort = SVector{n_ort}(1:n_ort)
#     idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))
#
#     w̄ = W.w_soc
#     v_idx = SVector{n_soc-1}(2:n_soc)
#     w̄0 = w̄[1]
#     w̄1 = w̄[v_idx]
#
#     # first solve
#     z_soc = z[idx_soc]
#     z0 = z_soc[1]
#     z1 = z_soc[v_idx]
#     ζ = dot(w̄1,z1)
#     Winvz = (1/W.η_soc)*[w̄0*z0 - ζ; z1 + (-z0 + ζ/(1 + w̄0))*w̄1]
#
#     # second solve
#     z0 = Winvz[1]
#     z1 = Winvz[v_idx]
#     ζ = dot(w̄1,z1)
#     W²invz = (1/W.η_soc)*[w̄0*z0 - ζ; z1 + (-z0 + ζ/(1 + w̄0))*w̄1]
#
#     [z[idx_ort] ./ (W.w_ort).^2 ; W²invz]
# end
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

function SA_block_diag(W1::Diagonal{T, SVector{n, T}}, W2::SMatrix{m,m,T,m2}) where {n,m,m2,T}
    # NOTE: this is only for testing
    top = hcat(W1, (@SMatrix zeros(n,m)))
    bot = hcat((@SMatrix zeros(m,n)), W2)
    vcat(top,bot)
end

# overload * and \ with the above functions
@inline function Base.:*(W::NT_lite{n_ort,n_soc,T}, z::SVector{n,T}) where {n_ort,n_soc,n,T}
    NT_vec_mul_lite(W::NT_lite{n_ort,n_soc,T}, z::SVector{n,T})
end
@inline function Base.:\(W::NT_lite{n_ort,n_soc,T}, z::SVector{n,T}) where {n_ort,n_soc,n,T}
    NT_lin_solve_lite(W::NT_lite{n_ort,n_soc,T}, z::SVector{n,T})
end
@inline function Base.:\(W::NT_lite{n_ort,n_soc,T}, Z::SMatrix{r,c,T,rc}) where {n_ort,n_soc,r,c,rc,T}
    NT_lin_solve_mat_lite(W::NT_lite{n_ort,n_soc,T}, Z::SMatrix{r,c,T,rc})
end
