using LinearAlgebra
using StaticArrays
using BenchmarkTools

@inline function soc_cone_product(u::SVector{n,T},v::SVector{n,T}) where {n,T}
    u0 = u[1]
    u1 = u[SVector{n-1}(2:n)]

    v0 = v[1]
    v1 = v[SVector{n-1}(2:n)]

    [dot(u,v);u0*v1 + v0*u1]
end

@inline function cone_product(s::SVector{n,T},z::SVector{n,T},idx_ort::SVector{n_ort,Ti}, idx_soc::SVector{n_soc, Ti}) where {n,T,n_ort,n_soc,Ti}
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))

    s_ort = s[idx_ort]
    z_ort = z[idx_ort]
    s_soc = s[idx_soc]
    z_soc = z[idx_soc]

    [s_ort .* z_ort; soc_cone_product(s_soc,z_soc)]
end

@inline function inverse_soc_cone_product(u::SVector{n,T},w::SVector{n,T}) where {n,T}
    u0 = u[1]
    u1 = u[SVector{n-1}(2:n)]

    w0 = w[1]
    w1 = w[SVector{n-1}(2:n)]

    ρ = u0^2 - dot(u1,u1)
    ν = dot(u1,w1)

    (1/ρ)*[u0*w0 - ν; (ν/u0 - w0)*u1 + (ρ/u0)*w1]
end

@inline function inverse_cone_product(λ::SVector{n,T},v::SVector{n,T},idx_ort::SVector{n_ort,Ti}, idx_soc::SVector{n_soc, Ti}) where {n,T,n_ort,n_soc,Ti}
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))

    λ_ort = λ[idx_ort]
    v_ort = v[idx_ort]
    λ_soc = λ[idx_soc]
    v_soc = v[idx_soc]

    top = v_ort ./ λ_ort
    bot = inverse_soc_cone_product(λ_soc,v_soc)
    [top;bot]
end

@inline function gen_e(idx_ort::SVector{n_ort,T}, idx_soc::SVector{n_soc,T}) where {n_ort, n_soc, T}
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))
    e1 = @SVector ones(n_ort)
    e2 = vcat(1.0, (@SVector zeros(n_soc - 1)))
    [e1;e2]
end
