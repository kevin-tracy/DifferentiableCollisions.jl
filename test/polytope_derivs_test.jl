
let
    path_str = joinpath(@__DIR__,"example_socps/polytopes.jld2")
    f = jldopen(path_str)
    A1 = SMatrix{14,3}(f["A1"])
    b1 = SVector{14}(f["b1"])
    A2 = SMatrix{8,3}(f["A2"])
    b2 = SVector{8}(f["b2"])

    P1 = DCD.Polytope(A1,b1)
    P2 = DCD.Polytope(A2,b2)

    function fd_ver(prim1,prim2,z;pdip_tol = 1e-10)
        idx_r1 = SVector{3}(1:3)
        idx_q1 = SVector{4}(4:7)
        idx_r2 = SVector{3}(8:10)
        idx_q2 = SVector{4}(11:14)
        prim1.r = z[idx_r1];
        prim1.q = z[idx_q1];
        prim2.r = z[idx_r2];
        prim2.q = z[idx_q2];
        α_star, x_star = DCD.proximity(prim1,prim2;pdip_tol)
        [x_star;α_star]
    end

    for i = 1:100
        P1.r = 2*(@SVector randn(3))
        P1.q = normalize((@SVector randn(4)))
        P2.r = 1*(@SVector randn(3))
        P2.q = normalize((@SVector randn(4)))




        my_tol = 1e-8

        α_star, x_star = DCD.proximity(P1,P2;pdip_tol = my_tol)
        α_star2, x_star2 = DCD.proximity(P2,P1; pdip_tol = my_tol)
        @test norm(α_star - α_star2) < my_tol*100
        @test norm(x_star - x_star2) < my_tol*100

        α2, x2, J = DCD.proximity_jacobian(P1,P2; verbose = false, pdip_tol = my_tol)
        J2 = FiniteDiff.finite_difference_jacobian(_z -> fd_ver(P1,P2,_z; pdip_tol = my_tol), [P1.r;P1.q;P2.r;P2.q])
        #
        @test norm(α_star - α2) < my_tol*100
        @test norm(x_star - x2) < my_tol*100
        @test norm(J-J2,Inf) < 1e-2

        if norm(J - J2) > 1
            @show P1.r
            @show P1.q
            @show P2.r
            @show P2.q
        end
    end

end
