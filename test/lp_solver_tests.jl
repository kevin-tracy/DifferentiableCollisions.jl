

function gen_lp(nx,ns)
    @assert ns > nx
    A = randn(ns,nx)
    b = randn(ns)
    c = [zeros(nx);ones(ns)]
    G = [A -I;-A -I]
    h = [b;-b]

    r1,c1 = size(G)

    nc = nx + ns
    return SVector{nc}(c), SMatrix{r1,c1}(G), SVector{r1}(h)
end

let

    for i = 1:100
        c,G,h= gen_lp(3,6)

        nh = length(h)
        idx_ort = SVector{nh}(1:nh)
        idx_soc1 = SArray{Tuple{0}, Int64, 1, 0}(())
        idx_soc2 = SArray{Tuple{0}, Int64, 1, 0}(())

        # x,s,z = DCD.solve_lp(c,G,h; pdip_tol = 1e-10, verbose = false)
        x,s,z = DCD.solve_socp(c,G,h,idx_ort,idx_soc1,idx_soc2; pdip_tol = 1e-10, verbose = false)

        r1 = G'z  + c
        r2 = s .* z
        r3 = G*x + s - h

        @test norm(r1) < 1e-8
        @test norm(r2) < 1e-8
        @test norm(r3) < 1e-8

        if i == 100
            @btime DCD.solve_socp($c,$G,$h,$idx_ort,$idx_soc1,$idx_soc2; pdip_tol = 1e-10, verbose = false)
        end
    end

end
