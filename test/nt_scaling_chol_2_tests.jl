let
    # create variables
    n_ort = 8
    n_soc1 = 4
    n_soc2 = 3
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc1 = SVector{n_soc1}((n_ort + 1):(n_ort + n_soc1))
    idx_soc2 = SVector{n_soc2}((n_ort + n_soc1 + 1):(n_ort + n_soc1 + n_soc2))

    ns = n_ort + n_soc1 + n_soc2
    s = SVector{ns}([abs.(randn(n_ort)); 10; abs.(randn(n_soc1 - 1));8;abs.(randn(n_soc2 - 1))])
    z = SVector{ns}([abs.(randn(n_ort)); 10; abs.(randn(n_soc1 - 1));8;abs.(randn(n_soc2 - 1))])

    W_nt = DCD.calc_NT_scalings(s,z,idx_ort,idx_soc1, idx_soc2)
    W_p1 = DCD.SA_block_diag(Diagonal(W_nt.ort), W_nt.soc1)
    W = Matrix(DCD.block_diag(W_p1, W_nt.soc2))

    # test

    # linear system solves W\b
    for i = 1:1000
        b = @SVector randn(n_ort + n_soc1 + n_soc2)
        x1 = W_nt\b
        x2 = W\b
        @test norm(x1 - x2) < 1e-13
    end

    for i = 1:1000
        # linear system solves W\B
        B = @SMatrix randn(n_ort + n_soc1 + n_soc2,4)
        x1 = W_nt\B
        x2 = W\B
        @test norm(x1 - x2) < 1e-13
    end

    for i = 1:1000
        # NT * vector
        g = @SVector randn(n_ort + n_soc1 + n_soc2)
        o1 = W_nt*g # overloades to NT_vec_mul
        o2 = W*g

        @test norm(o1 - o2) < 1e-13
    end

    for i = 1:1000
        # NT * vector
        G = @SMatrix randn(n_ort + n_soc1 + n_soc2, 4)
        o1 = W_nt * G
        o2 = W*G

        @test norm(o1 - o2) < 1e-13
    end

    # S_nt = DCD.scaling_2(s[idx_ort], DCD.arrow(s[idx_soc1]),DCD.arrow(s[idx_soc2]))
    Z_nt = DCD.scaling_2(z[idx_ort], DCD.arrow(z[idx_soc1]),DCD.arrow(z[idx_soc2]))


    # S_p1 = DCD.SA_block_diag(Diagonal(s[idx_ort]), DCD.arrow(s[idx_soc1]))
    # S = Matrix(DCD.block_diag(S_p1, DCD.arrow(s[idx_soc2])))

    Z_p1 = DCD.SA_block_diag(Diagonal(z[idx_ort]), DCD.arrow(z[idx_soc1]))
    Z = Matrix(DCD.block_diag(Z_p1, DCD.arrow(z[idx_soc2])))

    # A = S_nt\Z_nt
    for i = 1:1000
        # NT * vector
        G = @SMatrix randn(n_ort + n_soc1 + n_soc2, 4)
        o1 = (W\Z) * G
        o2 = (W_nt\Z_nt)*G
    #
        @test norm(o1 - o2) < 1e-13
    end

end
