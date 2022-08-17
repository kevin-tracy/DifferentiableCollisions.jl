using LinearAlgebra
using StaticArrays
using BenchmarkTools


function gen_lp(nx,ns)
    x = randn(nx);
    @show x
    G = randn(ns,nx)
    s = abs.(randn(ns));
    z = abs.(randn(ns));
    # this encodes which constraints are active
    for i = 1:ns
        if randn() < 0.5
            s[i] = 0 ;
        else
            z[i] = 0 ;
        end
    end
    h = G*x + s;
    q = -G'*z;

    nxns = nx*ns
    return SVector{nx}(q), SMatrix{ns,nx}(G), SVector{ns}(h)
end


include("/Users/kevintracy/.julia/dev/DCD/src/lp_solver.jl")

@inline function pdip_init_2(q::SVector{nx,T},
                             G::SMatrix{ns,nx,T,ns_nx},
                             h::SVector{ns,T}) where {nx,T,ns,ns_nx}
    # initialization for PDIP
    # K = Symmetric(G'*G)
    # F = cholesky(K)
    # x = F\(G'*h-q)
    # z = G*x - h
    # s = 1*z
    # α_p = -minimum(-z)
    # if α_p < 0
    #     s = -z
    # else
    #     s = -z .+ (1 + α_p)
    # end
    # α_d = -minimum(z)
    # if α_d >= 0
    #     z = z .+ (1 + α_d)
    # end
    # return x,s,z
end

let

    c,G,h = gen_lp(3,6)
    x,s,z = pdip(c,G,h; tol = 1e-6, verbose = true)
    # @show x
    # @show typeof(c)
    # @show typeof(G)
    # @show typeof(h)
    #
    # @btime pdip_init_2($c,$G,$h)



    @btime pdip($c,$G,$h; tol = 1e-6, verbose = false)

end
