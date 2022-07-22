function kkt_R(capsule::DCD.Capsule{T},
           cone::DCD.Cone{T},
           x,
           z,
           r1,
           q1,
           r2,
           q2,
           idx_ort::SVector{n_ort,Ti},
           idx_soc1::SVector{n_soc1,Ti},
           idx_soc2::SVector{n_soc2,Ti}) where {T,nx,nz,n_ort,n_soc1,n_soc2,Ti}

    G_ort1, h_ort1, G_soc1, h_soc1 = DCD.problem_matrices(capsule,r1,q1)
    G_ort2, h_ort2, G_soc2, h_soc2 = DCD.problem_matrices(cone,r2,q2)

    n_ort1 = length(h_ort1)
    n_ort2 = length(h_ort2)

    G_ort_top = G_ort1
    G_ort_bot = hcat(G_ort2, (@SVector zeros(n_ort2))) # add a column for γ (capsule)

    G_soc_top = G_soc1
    G_soc_bot = hcat(G_soc2, (@SVector zeros(n_soc2))) # add a column for γ (capsule)

    G_ = [G_ort_top;G_ort_bot;G_soc_top;G_soc_bot]
    h_ = [h_ort1;h_ort2;h_soc1;h_soc2]


    c = SA[0,0,0,1.0,0]

    [
    c + G_'*z;
    DCD.cone_product(h_ - G_*x, z, idx_ort, idx_soc1, idx_soc2)
    ]
end

function solve_alpha(capsule::DCD.Capsule{T},
           cone::DCD.Cone{T},
           r1,
           q1,
           r2,
           q2,
           idx_ort::SVector{n_ort,Ti},
           idx_soc1::SVector{n_soc1,Ti},
           idx_soc2::SVector{n_soc2,Ti}) where {T,n_ort,n_soc1,n_soc2,Ti}

    G_ort1, h_ort1, G_soc1, h_soc1 = DCD.problem_matrices(capsule,r1,q1)
    G_ort2, h_ort2, G_soc2, h_soc2 = DCD.problem_matrices(cone,r2,q2)

    n_ort1 = length(h_ort1)
    n_ort2 = length(h_ort2)

    G_ort_top = G_ort1
    G_ort_bot = hcat(G_ort2, (@SVector zeros(n_ort2))) # add a column for γ (capsule)

    G_soc_top = G_soc1
    G_soc_bot = hcat(G_soc2, (@SVector zeros(n_soc2))) # add a column for γ (capsule)

    G_ = [G_ort_top;G_ort_bot;G_soc_top;G_soc_bot]
    h_ = [h_ort1;h_ort2;h_soc1;h_soc2]


    x,s,z = DCD.solve_socp(SA[0,0,0,1.0,0],G_,h_,idx_ort,idx_soc1,idx_soc2; verbose = false, pdip_tol = 1e-12)
    [x[4]]
end

let

    cone = DCD.Cone(2.0,deg2rad(22))
    cone.r = 0.3*(@SVector randn(3))
    cone.q = normalize((@SVector randn(4)))

    capsule = DCD.Capsule(.3,1.2)
    capsule.r = (@SVector randn(3))
    capsule.q = normalize((@SVector randn(4)))

    G_ort1, h_ort1, G_soc1, h_soc1 = DCD.problem_matrices(capsule)
    G_ort2, h_ort2, G_soc2, h_soc2 = DCD.problem_matrices(cone)

    n_ort1_ = length(h_ort1)
    n_ort2_ = length(h_ort2)
    n_soc1_ = length(h_soc1)
    n_soc2_ = length(h_soc2)
    n_ort_ = n_ort1_ + n_ort2_

    G_ort_top = G_ort1
    G_ort_bot = hcat(G_ort2, (@SVector zeros(n_ort2_))) # add a column for γ (capsule)

    G_soc_top = G_soc1
    G_soc_bot = hcat(G_soc2, (@SVector zeros(n_soc2_))) # add a column for γ (capsule)

    G_ = [G_ort_top;G_ort_bot;G_soc_top;G_soc_bot]
    h_ = [h_ort1;h_ort2;h_soc1;h_soc2]

    idx_ort = SVector{n_ort_}(1:n_ort_)
    idx_soc1 = SVector{n_soc1_}((n_ort_ + 1):(n_ort_ + n_soc1_))
    idx_soc2 = SVector{n_soc2_}((n_ort_ + n_soc1_ + 1):(n_ort_ + n_soc1_ + n_soc2_))

    x,s,z = DCD.solve_socp(SA[0,0,0,1.0,0],G_,h_,idx_ort,idx_soc1,idx_soc2; verbose = true, pdip_tol = 1e-10)

    cone_prod_res= DCD.cone_product(h_ - G_*x,z,idx_ort,idx_soc1,idx_soc2)
    @test norm(cone_prod_res) < 1e-3

    res =  kkt_R(capsule,cone,x,z,capsule.r,capsule.q,cone.r,cone.q,idx_ort,idx_soc1,idx_soc2)
    @test norm(res) < 1e-3

    nx = length(x); nz = length(z)
    idx_x = SVector{nx}(1:length(x))
    idx_z = SVector{nz}((length(x) + 1):(length(x) + length(z)))
    dR_dw=FiniteDiff.finite_difference_jacobian(_w -> kkt_R(capsule,cone,_w[idx_x],_w[idx_z],capsule.r,capsule.q,cone.r,cone.q,idx_ort,idx_soc1,idx_soc2),[x;z])
    idx_r1 = SVector{3}(1:3)
    idx_q1 = SVector{4}(4:7)
    idx_r2 = SVector{3}(8:10)
    idx_q2 = SVector{4}(11:14)
    dR_dθ=FiniteDiff.finite_difference_jacobian(_θ -> kkt_R(capsule,cone,x,z,_θ[idx_r1],_θ[idx_q1],_θ[idx_r2],_θ[idx_q2],idx_ort,idx_soc1,idx_soc2), [capsule.r;capsule.q;cone.r;cone.q])

    dw_dθ = -dR_dw\dR_dθ

    dα_dθ=FiniteDiff.finite_difference_jacobian(_θ -> solve_alpha(capsule,cone,_θ[idx_r1],_θ[idx_q1],_θ[idx_r2],_θ[idx_q2],idx_ort,idx_soc1,idx_soc2), [capsule.r;capsule.q;cone.r;cone.q])

    @show norm(vec(dα_dθ) - vec(dw_dθ[4,:]))
    @test norm(vec(dα_dθ) - vec(dw_dθ[4,:])) < 1e-4


end
