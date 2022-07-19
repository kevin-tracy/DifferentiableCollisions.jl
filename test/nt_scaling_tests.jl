
function test_ort_nt_scaling(s,z,idx_ort)
    s_ort = s[idx_ort]
    z_ort = z[idx_ort]
    Diagonal(sqrt.(s_ort ./ z_ort))
end
function test_normalize_soc(x)
    nx = length(x)
    J = Diagonal([1;-ones(nx-1)])
    x̄ = x*(1/sqrt(x'*J*x))
end

function test_soc_nt_scaling(s,z,idx_soc)
    s_soc = s[idx_soc]
    z_soc = z[idx_soc]
    nx = length(s_soc)
    J = Diagonal([1;-ones(nx-1)])
    z̄ = test_normalize_soc(z_soc)
    s̄ = test_normalize_soc(s_soc)
    γ = sqrt((1 + dot(z̄,s̄))/2)
    w̄ = (1/(2*γ))*(s̄ + J*z̄)
    b = (1/(w̄[1] + 1))
    W̄ = [w̄'; w̄[2:end] (I + b*w̄[2:end]*w̄[2:end]')]
    W = W̄*((s_soc'*J*s_soc)/(z_soc'*J*z_soc))^(1/4)
end
function test_nt_scaling(s,z,idx_ort, idx_soc)
    W1 = test_ort_nt_scaling(s,z,idx_ort)
    W2 = test_soc_nt_scaling(s,z,idx_soc)
    Matrix(blockdiag(sparse(W1),sparse(W2)))
end

function test_conelp_nt_scaling(s,z)
    # %% check whether s and z are in the cone.
    s0 = s[1]; s1 = s[2:end];
    z0 = z[1]; z1 = z[2:end];

    sres = s0^2 - s1'*s1;
    zres = z0^2 - z1'*z1;

    # assert( sres > 0, 's not in second-order cone');
    # assert( zres > 0, 'z not in second-order cone');
    @assert sres > 0
    @assert zres > 0


    # %% scalings
    sbar = s ./ sqrt(sres);
    zbar = z ./ sqrt(zres);
    eta = (sres / zres)^(1/4);
    gamma = sqrt( (1+sbar'*zbar)/2 );
    wbar = 1/2/gamma*(sbar + [zbar[1]; -zbar[2:end]]);
    q = wbar[2:end];
    a = wbar[1];
    b = 1/(1+a);
    W = eta*[a q'; q ( diagm(ones(length(q))) + b*(q*q'))];
    return W
end

let
    # create variables
    n_ort = 8
    n_soc = 4
    idx_ort = SVector{n_ort}(1:n_ort)
    idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))

    ns = n_ort + n_soc
    s = SVector{ns}([abs.(randn(n_ort)); 10; abs.(randn(n_soc - 1))])
    z = SVector{ns}([abs.(randn(n_ort)); 10; abs.(randn(n_soc - 1))])

    W_ort, W_ort_inv = DCD.ort_nt_scaling(s[idx_ort], z[idx_ort])
    W_soc, W_soc_inv = DCD.soc_nt_scaling(s[idx_soc], z[idx_soc])
    W_soc_ecos = test_conelp_nt_scaling(s[idx_soc],z[idx_soc])

    @test norm(W_soc - W_soc_ecos) < 1e-13
    @test norm(W_ort*W_ort_inv - I) < 1e-13
    @test norm(W_soc*W_soc_inv - I) < 1e-13

    # compare the NT scaling real matrices
    W1 = DCD.SA_block_diag(W_ort,W_soc)
    W2 = test_nt_scaling(s,z,idx_ort, idx_soc)
    @test norm(W1 - W2) < 1e-13

    # use our NT type
    W = DCD.NT(W_ort, W_soc, W_ort_inv, W_soc_inv)

    # NT * NT
    W_squared = W1*W1
    W² = DCD.NT(W_ort^2, W_soc^2, W_ort_inv^2, W_soc_inv^2)
    # DCD.square_NT!(W²)
    # W_squared_2 = DCD.SA_block_diag(W².W_ort, W².W_soc)
    #
    W_squared_inv = inv(W_squared)
    W_squared_inv2 = DCD.SA_block_diag(W².W_ort_inv, W².W_soc_inv)
    @test norm(W_squared_inv - W_squared_inv2) < 1e-13

    # linear system solves W\b
    b = @SVector randn(n_ort + n_soc)
    x1 = W1\b
    x2 = W\b
    @test norm(x1 - x2) < 1e-13

    # linear system solves W²\b
    x1 = W1\(W1\b)
    x2 = W\(W\b)
    x3 = W²\b
    x4 = W_squared\b
    @test norm(x1 - x2) < 1e-13
    @test norm(x1 - x3) < 1e-13
    @test norm(x1 - x4) < 1e-13

    # linear system solves W\B
    B = @SMatrix randn(n_ort + n_soc,4)
    x1 = W1\B
    x2 = W\B
    @test norm(x1 - x2) < 1e-13

    # linear system solves W²\b
    x1 = W1\(W1\B)
    x2 = W\(W\B)
    x3 = W²\B
    x4 = W_squared\B
    @test norm(x1 - x2) < 1e-13
    @test norm(x1 - x3) < 1e-13
    @test norm(x1 - x4) < 1e-13

    # NT * matrix
    G = @SMatrix randn(n_soc + n_ort, 4)
    O = W1*G
    O2 = [W_ort*G[idx_ort,:]; W_soc*G[idx_soc,:]]
    O3 = W*G # overloads to NT_matmul

    @test norm(O - O2) < 1e-13
    @test norm(O - O3) < 1e-13

    # NT * vector
    g = @SVector randn(n_soc + n_ort)
    o1 = W*g # overloades to NT_vec_mul
    o2 = W1*g

    @test norm(o1 - o2) < 1e-13

    # @btime DCD.ort_nt_scaling($s[$idx_ort], $z[$idx_ort])
    # @btime DCD.soc_nt_scaling($s[$idx_soc], $z[$idx_soc])
    # @btime $W*$G

    # @btime $W\$b
    # @btime vcat($W.W_ort_inv*$b[$idx_ort], $W.W_soc_inv*$b[$idx_soc])

    # s_soc = s[idx_soc]
    # z_soc = z[idx_soc]
    #
    # @btime soc_nt_scaling_2($s_soc,$z_soc)
    # @btime DCD.soc_nt_scaling($s_soc,$z_soc)
end
