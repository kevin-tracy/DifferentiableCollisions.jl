using LinearAlgebra
using StaticArrays
using JLD2

# struct ConeProg1{nx,ns,nsnx,n_ort,n_soc,Tf,Ti}
#     c::SVector{nx,Tf}
#     G::SMatrix{ns,nx,Tf,nsnx}
#     h::SVector{ns,Tf}
#     idx_ort::SVector{n_ort, Ti}
#     idx_soc::SVector{n_soc, Ti}
# end


function build_pr()
    @load "/Users/kevintracy/.julia/dev/DCD/extras/example_socp.jld2"

    nx = 5
    n_ort = length(h_ort)
    n_soc = length(h_soc)

    G = SMatrix{n_ort + n_soc, nx}([G_ort;G_soc])
    h = SVector{n_ort + n_soc}([h_ort;h_soc])

    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))

    # nx = 5
    # ns = n_ort + n_soc
    # nz = ns
    #
    # idx_x = 1:nx
    # idx_s = (nx + 1):(nx + ns)
    # idx_z = (nx + ns + 1):(nx + ns + nz)


    c = SA[0,0,0,1,0.0]

    θ = ConeProg1(c,G,h,idx_ort,idx_soc)

    @btime θ = ConeProg1($c,$G,$h,$idx_ort,$idx_soc)
    @btime p = (c = $c, G = $G, h = $h, idx_ort = $idx_ort, idx_soc = $idx_soc)
    @show typeof(θ)

    # θ = (G = G, h = h, c = c, idx_ort = idx_ort, idx_soc = idx_soc,
         # n_ort = n_ort, n_soc = n_soc, nx = nx, ns = ns, nz = nz,
         # idx_x = idx_x, idx_s = idx_s, idx_z = idx_z)

         @btime test_pr($θ)
end

function test_pr(θ::ConeProg1{nx,ns,nsnx,n_ort,n_soc,Tf,Ti}) where {nx,ns,nsnx,n_ort,n_soc,Tf,Ti}
    a =  n_ort
end

build_pr()
