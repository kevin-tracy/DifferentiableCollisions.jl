



function test_arrow(v)
    [v'; v[2:end] v[1]*I]
end
function test_inverse_soc_cone_prod(λ_soc,v_soc)
    test_arrow(λ_soc)\v_soc
end
function test_soc_prod(u,v)
    u0 = u[1]
    u1 = u[2:end]

    v0 = v[1]
    v1 = v[2:end]

    [dot(u,v);u0*v1 + v0*u1]
end

# create variables
n_ort = 8
n_soc = 4
idx_ort = SVector{n_ort}(1:n_ort)
idx_soc = SVector{n_soc}((n_ort + 1):(n_ort + n_soc))

ns = n_ort + n_soc
s = SVector{ns}([abs.(randn(n_ort)); 10; abs.(randn(n_soc - 1))])
z = SVector{ns}([abs.(randn(n_ort)); 10; abs.(randn(n_soc - 1))])

s_soc = s[idx_soc]
z_soc = z[idx_soc]
o1 = test_soc_prod(s_soc,z_soc)
o2 = DCD.soc_cone_product(s_soc,z_soc)

@test norm(o1-o2) < 1e-13

o1 = test_inverse_soc_cone_prod(s_soc,z_soc)
o2 = DCD.inverse_soc_cone_product(s_soc,z_soc)

@test norm(o1-o2) < 1e-13

# e = DCD.gen_e(idx_ort, idx_soc)
#
# @show e
# @show typeof(e)
# @btime DCD.gen_e($idx_ort, $idx_soc)

# @btime DCD.soc_cone_product($s_soc,$z_soc)
# @btime DCD.inverse_soc_cone_product($s_soc,$z_soc)
