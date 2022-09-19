
# function build_pr_1()
#     path_str = joinpath(@__DIR__,"example_socps/example_socp.jld2")
#     f = jldopen(path_str)
#     G_ort = f["G_ort"]
#     h_ort = f["h_ort"]
#     G_soc = f["G_soc"]
#     h_soc = f["h_soc"]
#
#
#     nx = 5
#     n_ort = length(h_ort)
#     n_soc = length(h_soc)
#
#     G = SMatrix{n_ort + n_soc, nx}([G_ort;G_soc])
#     h = SVector{n_ort + n_soc}([h_ort;h_soc])
#
#     idx_ort = SVector{n_ort}(1:n_ort)
#     idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))
#
#     c = SA[0,0,0,1,0.0]
#
#     return c, G, h, idx_ort, idx_soc
# end

# let
#
#     c,G,h,idx_ort,idx_soc = build_pr_1()
#
#     x,s,z = DCD.solve_socp(c,G,h,idx_ort,idx_soc;verbose = true, pdip_tol = 1e-12)
#     @test abs(dot(s,z))<1e-10
#
#     @btime DCD.solve_socp($c,$G,$h,$idx_ort,$idx_soc; verbose = false)
#
# end


function build_pr_2()
    # @load "/Users/kevintracy/.julia/dev/DCD/extras/example_socp_2.jld2"
    path_str = joinpath(@__DIR__,"example_socps/example_socp_2.jld2")
    f = jldopen(path_str)
    G = f["G"]
    h = f["h"]
    n_ort = f["n_ort"]
    n_soc1 = f["n_soc1"]
    n_soc2 = f["n_soc2"]

    nx = 5
    ns = n_ort + n_soc1 + n_soc2
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc1 = SVector{n_soc1}((n_ort + 1):(n_ort + n_soc1))
    idx_soc2 = SVector{n_soc2}((n_ort + n_soc1 + 1):(n_ort + n_soc1 + n_soc2))

    c = SA[0,0,0,1,0.0]

    return c, SMatrix{ns,nx}(G), SVector{ns}(h), idx_ort, idx_soc1, idx_soc2
end

let

    c,G,h,idx_ort,idx_soc1,idx_soc2 = build_pr_2()

    x,s,z = DCD.solve_socp(c,G,h,idx_ort,idx_soc1,idx_soc2;verbose = true, pdip_tol = 1e-12)
    @test abs(dot(s,z))<1e-10
    # @btime DCD.solve_socp($c,$G,$h,$idx_ort,$idx_soc1,$idx_soc2; verbose = false)

end
