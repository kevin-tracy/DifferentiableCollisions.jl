using LinearAlgebra
using BenchmarkTools
mutable struct LUSolver{T}
    A::Array{T,2}
    ipiv::Vector{Int}
    lda::Int
    info::Base.RefValue{Int}
end

function lu_solver(A)
    m, n = size(A)
    ipiv = similar(A, LinearAlgebra.BlasInt, min(m, n))
    lda  = max(1, stride(A, 2))
    info = Ref{LinearAlgebra.BlasInt}()
    LUSolver(copy(A), ipiv, lda, info)
end

function getrf!(A, ipiv, lda, info)
    Base.require_one_based_indexing(A)
    LinearAlgebra.chkstride1(A)
    m, n = size(A)
    lda  = max(1,stride(A, 2))
    ccall((LinearAlgebra.BLAS.@blasfunc(dgetrf_), Base.liblapack_name), Cvoid,
          (Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ptr{Float64},
           Ref{LinearAlgebra.BlasInt}, Ptr{LinearAlgebra.BlasInt}, Ptr{LinearAlgebra.BlasInt}),
          m, n, A, lda, ipiv, info)
    return nothing
end

function factorize!(s::LUSolver{T}, A::AbstractMatrix{T}) where T
    fill!(s.A, 0.0)
    fill!(s.ipiv, 0)
    s.lda = 0
    s.A .= A
    getrf!(s.A, s.ipiv, s.lda, s.info)
end

function linear_solve!(s::LUSolver{T}, x::Vector{T}, A::Matrix{T},
        b::Vector{T}; reg::T = 0.0, fact::Bool = true) where T
    fact && factorize!(s, A)
    x .= b
    LinearAlgebra.LAPACK.getrs!('N', s.A, s.ipiv, x)
end

function linear_solve!(s::LUSolver{T}, x::Matrix{T}, A::Matrix{T},
    b::Matrix{T}; reg::T = 0.0, fact::Bool = true) where T
    fill!(x, 0.0)
    n, m = size(x)
    r_idx = 1:n
    fact && factorize!(s, A)
    x .= b
    for j = 1:m
        xv = @views x[r_idx, j]
        LinearAlgebra.LAPACK.getrs!('N', s.A, s.ipiv, xv)
    end
end

let
    A = randn(10,10)

    s = lu_solver(A)

    factorize!(s,A)
    @btime factorize!($s,$A)
    B = randn(10,6)
    X = 0*B

    @btime linear_solve!($s,$X,$A,$B; fact = true)

end
