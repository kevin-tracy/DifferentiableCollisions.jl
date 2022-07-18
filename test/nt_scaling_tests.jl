

n_ort = 8
n_soc = 4
idx_ort = SVector{n_ort}(1:n_ort)
idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))

ns = n_ort + n_soc
s = SVector{ns}([abs.(randn(n_ort)); 10; abs.(randn(n_soc - 1))])
z = SVector{ns}([abs.(randn(n_ort)); 10; abs.(randn(n_soc - 1))])

W_ort = DCD.ort_nt_scaling(s[idx_ort], z[idx_ort])
W_soc = DCD.soc_nt_scaling(s[idx_soc], z[idx_soc])

# W1 = NT(W_ort, W_soc)
W = DCD.SA_block_diag(W_ort,W_soc)

function ort_nt_scaling(s,z,idx_ort)
    s_ort = s[idx_ort]
    z_ort = z[idx_ort]
    Diagonal(sqrt.(s_ort ./ z_ort))
end
function normalize_soc(x)
    nx = length(x)
    J = Diagonal([1;-ones(nx-1)])
    x̄ = x*(1/sqrt(x'*J*x))
end
function soc_nt_scaling(s,z,θ)
    s_soc = s[idx_soc]
    z_soc = z[idx_soc]
    nx = length(s_soc)
    J = Diagonal([1;-ones(nx-1)])
    z̄ = normalize_soc(z_soc)
    s̄ = normalize_soc(s_soc)
    γ = sqrt((1 + dot(z̄,s̄))/2)
    w̄ = (1/(2*γ))*(s̄ + J*z̄)
    b = (1/(w̄[1] + 1))
    W̄ = [w̄'; w̄[2:end] (I + b*w̄[2:end]*w̄[2:end]')]
    W = W̄*((s_soc'*J*s_soc)/(z_soc'*J*z_soc))^(1/4)
end
function nt_scaling(s,z,idx_ort, idx_soc)
    W1 = ort_nt_scaling(s,z,idx_ort)
    W2 = soc_nt_scaling(s,z,idx_soc)
    Matrix(blockdiag(sparse(W1),sparse(W2)))
end

W2 = nt_scaling(s,z,idx_ort, idx_soc)

W_ort_2 = W2[idx_ort, idx_ort]
W_soc_2 = W2[idx_soc, idx_soc]

@test norm(W_ort - W_ort_2) < 1e-13
@test norm(W_soc - W_soc_2) < 1e-13

# @btime DCD.ort_nt_scaling($s[$idx_ort], $z[$idx_ort])
# @btime DCD.soc_nt_scaling($s[$idx_soc], $z[$idx_soc])
