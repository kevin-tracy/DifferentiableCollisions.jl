
@inline function arrow(x::SVector{n,T}) where {n,T}
    if n > 0
        top = x'
        bot = [x[SVector{n-1}(2:n)] x[1]*diagm((@SVector ones(n-1)))]
        return vcat(top,bot)
    else
        return SArray{Tuple{0,0}, Float64, 2, 0}(())
    end
end

@inline function soc_cone_product(u::SVector{n,T1},v::SVector{n,T2}) where {n,T1,T2}
    if n > 0
        u0 = u[1]
        u1 = u[SVector{n-1}(2:n)]

        v0 = v[1]
        v1 = v[SVector{n-1}(2:n)]

        return [dot(u,v);u0*v1 + v0*u1]
    else
        return SArray{Tuple{0}, Float64, 1, 0}(())
    end
end

# @inline function cone_product(s::SVector{n,T1},z::SVector{n,T2},idx_ort::SVector{n_ort,Ti}, idx_soc::SVector{n_soc, Ti}) where {n,T1,T2,n_ort,n_soc,Ti}
#     idx_ort = SVector{n_ort}(1:n_ort)
#     idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))
#
#     s_ort = s[idx_ort]
#     z_ort = z[idx_ort]
#     s_soc = s[idx_soc]
#     z_soc = z[idx_soc]
#
#     [s_ort .* z_ort; soc_cone_product(s_soc,z_soc)]
# end

@inline function cone_product(s::SVector{n,T1},z::SVector{n,T2},idx_ort::SVector{n_ort,Ti}, idx_soc1::SVector{n_soc1, Ti},idx_soc2::SVector{n_soc2, Ti}) where {n,T1,T2,n_ort,n_soc1,n_soc2,Ti}
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc1 = SVector{n_soc1}((n_ort + 1):(n_ort + n_soc1))
    idx_soc2 = SVector{n_soc2}((n_ort + n_soc1 + 1):(n_ort + n_soc1 + n_soc2))

    s_ort = s[idx_ort]
    z_ort = z[idx_ort]
    s_soc1 = s[idx_soc1]
    z_soc1 = z[idx_soc1]
    s_soc2 = s[idx_soc2]
    z_soc2 = z[idx_soc2]

    [s_ort .* z_ort; soc_cone_product(s_soc1,z_soc1);soc_cone_product(s_soc2,z_soc2)]
end
@inline function inverse_soc_cone_product(u::SVector{n,T1},w::SVector{n,T2}) where {n,T1,T2}
    if n > 0
        u0 = u[1]
        u1 = u[SVector{n-1}(2:n)]

        w0 = w[1]
        w1 = w[SVector{n-1}(2:n)]

        ρ = u0^2 - dot(u1,u1)
        ν = dot(u1,w1)

        return (1/ρ)*[u0*w0 - ν; (ν/u0 - w0)*u1 + (ρ/u0)*w1]
    else
        return SArray{Tuple{0}, Float64, 1, 0}(())
    end
end

# @inline function inverse_cone_product(λ::SVector{n,T1},v::SVector{n,T2},idx_ort::SVector{n_ort,Ti}, idx_soc::SVector{n_soc, Ti}) where {n,T1,T2,n_ort,n_soc,Ti}
#     idx_ort = SVector{n_ort}(1:n_ort)
#     idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))
#
#     λ_ort = λ[idx_ort]
#     v_ort = v[idx_ort]
#     λ_soc = λ[idx_soc]
#     v_soc = v[idx_soc]
#
#     top = v_ort ./ λ_ort
#     bot = inverse_soc_cone_product(λ_soc,v_soc)
#     [top;bot]
# end

@inline function inverse_cone_product(λ::SVector{n,T1},v::SVector{n,T2},idx_ort::SVector{n_ort,Ti}, idx_soc1::SVector{n_soc1, Ti},idx_soc2::SVector{n_soc2, Ti}) where {n,T1,T2,n_ort,n_soc1,n_soc2,Ti}
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc1 = SVector{n_soc1}((n_ort + 1):(n_ort + n_soc1))
    idx_soc2 = SVector{n_soc2}((n_ort + n_soc1 + 1):(n_ort + n_soc1 + n_soc2))

    λ_ort = λ[idx_ort]
    v_ort = v[idx_ort]
    λ_soc1 = λ[idx_soc1]
    v_soc1 = v[idx_soc1]
    λ_soc2 = λ[idx_soc2]
    v_soc2 = v[idx_soc2]

    top = v_ort ./ λ_ort
    bot1 = inverse_soc_cone_product(λ_soc1,v_soc1)
    bot2 = inverse_soc_cone_product(λ_soc2,v_soc2)
    [top;bot1;bot2]
end

# @inline function gen_e(idx_ort::SVector{n_ort,T}, idx_soc::SVector{n_soc,T}) where {n_ort, n_soc, T}
#     idx_ort = SVector{n_ort}(1:n_ort)
#     idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))
#     e1 = @SVector ones(n_ort)
#     e2 = vcat(1.0, (@SVector zeros(n_soc - 1)))
#     [e1;e2]
# end
@inline function gen_e(idx_ort::SVector{n_ort,T},
                       idx_soc1::SVector{n_soc1,T},
                       idx_soc2::SVector{n_soc2,T}) where {n_ort, n_soc1,n_soc2,T}
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc1 = SVector{n_soc1}((n_ort + 1):(n_ort + n_soc1))
    idx_soc2 = SVector{n_soc2}((n_ort + n_soc1 + 1):(n_ort + n_soc1 + n_soc2))

    e = @SVector ones(n_ort)
    if n_soc1>0
        e = [e;vcat(1.0, (@SVector zeros(n_soc1 - 1)))]
    end
    if n_soc2>0
        e = [e;vcat(1.0, (@SVector zeros(n_soc2 - 1)))]
    end
    return e

    # e2 = vcat(1.0, (@SVector zeros(n_soc1 - 1)))
    # e3 = vcat(1.0, (@SVector zeros(n_soc2 - 1)))
    # [e1;e2;e3]
end

@inline function normalize_soc(x::SVector{nx,T}) where {nx, T}
    x/sqrt(soc_quad_J(x))
end
@inline function soc_quad_J(x::SVector{nx,T}) where {nx,T}
    xs = x[1]
    xv = x[SVector{nx-1}(2:nx)]
    xs^2 - dot(xv,xv)
end

function SA_block_diag(W1::Diagonal{T, SVector{n, T}}, W2::SMatrix{m,m,T,m2}) where {n,m,m2,T}
    # NOTE: this is only for testing
    top = hcat(W1, (@SMatrix zeros(n,m)))
    bot = hcat((@SMatrix zeros(m,n)), W2)
    vcat(top,bot)
end
function block_diag(W1, W2)
    # NOTE: this is only for testing
    n = size(W1,1)
    m = size(W2,1)
    top = hcat(W1, (@SMatrix zeros(n,m)))
    bot = hcat((@SMatrix zeros(m,n)), W2)
    vcat(top,bot)
end
