function test_stack_two(G1,G2)
    r1,v1 = size(G1)
    r2,v2 = size(G2)

    v1_extra = v1 - 4
    v2_extra = v2 - 4
    v1_extra_idx = (v1_extra == 0) ? [] : (5:(v1_extra + 4))
    v2_extra_idx = (v2_extra == 0) ? [] : (5:(v2_extra + 4))


    G1_p = G1[:,1:4]
    G2_p = G2[:,1:4]

    G1_e = G1[:,v1_extra_idx]
    G2_e = G2[:,v2_extra_idx]

    [
    G1_p G1_e               zeros(r1,v2_extra);
    G2_p zeros(r2,v1_extra) G2_e
    ]
end
let

    n_ort1 = 2
    n_ort2 = 5
    n_soc1 = 3
    n_soc2 = 4

    v1s = [6,4,4,4,7,6]
    v2s = [6,4,5,5,6,7]
    for i = 1:length(v1s)

        v1 = v1s[i]
        v2 = v2s[i]

        G_ort1 = @SMatrix randn(n_ort1,v1)
        h_ort1 = @SVector randn(n_ort1)
        G_ort2 = @SMatrix randn(n_ort2,v2)
        h_ort2 = @SVector randn(n_ort2)

        G_soc1 = @SMatrix randn(n_soc1,v1)
        h_soc1 = @SVector randn(n_soc1)
        G_soc2 = @SMatrix randn(n_soc2,v2)
        h_soc2 = @SVector randn(n_soc2)

        c,G,h,idx_ort,idx_soc1,idx_soc2 = DCD.combine_problem_matrices(G_ort1,h_ort1,G_soc1, h_soc1, G_ort2,h_ort2, G_soc2, h_soc2)
        # @btime combine_problem_matrices($G_ort1,$h_ort1,$G_soc1, $h_soc1, $G_ort2,$h_ort2, $G_soc2, $h_soc2)
        G2 = [test_stack_two(G_ort1,G_ort2);test_stack_two(G_soc1,G_soc2)]
        h2 = [h_ort1;h_ort2;h_soc1;h_soc2]
        @test norm(G-G2) < 1e-14
        @test norm(h - h2) < 1e-14
        @test size(G2,2) == length(c)
        @test size(G2,1) == length(h2)

    end

end
