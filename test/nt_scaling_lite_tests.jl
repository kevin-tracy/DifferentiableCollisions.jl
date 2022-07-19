
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

@test norm(W_soc - W_soc_ecos) < 1e-13
@test norm(W_ort*W_ort_inv - I) < 1e-13
@test norm(W_soc*W_soc_inv - I) < 1e-13

# compare the NT scaling real matrices
W = DCD.SA_block_diag(W_ort,W_soc)

# use our NT type
Wnt = DCD.NT(W_ort, W_soc, W_ort_inv, W_soc_inv)

w_ort        = DCD.ort_nt_scaling_lite(s[idx_ort],z[idx_ort])
w_soc, η_soc = DCD.soc_nt_scaling_lite(s[idx_soc],z[idx_soc])
Wnt_lite = DCD.NT_lite(w_ort, w_soc, η_soc)

b = @SVector randn(n_ort + n_soc)

p1 = W * b
p2 = Wnt * b
p3 = Wnt_lite * b

@test norm(p1 - p2) < 1e-13
@test norm(p1 - p3) < 1e-13
