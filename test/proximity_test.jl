let
    cone = DCD.Cone(2.0,deg2rad(22))
    cone.r = 0.3*(@SVector randn(3))
    cone.q = normalize((@SVector randn(4)))

    capsule = DCD.Capsule(.3,1.2)
    capsule.r = (@SVector randn(3))
    capsule.q = normalize((@SVector randn(4)))

    # α, x = DCD.proximity(capsule,cone)
    # @btime DCD.proximity($capsule,$cone)
    α, x = DCD.proximity(cone,capsule)
    α2,x2 = DCD.proximity(capsule,cone)
    @test abs(α - α2) < 1e-4
    @test norm(x - x2) < 1e-4
    # @btime DCD.proximity($capsule,$cone)
    # @info α

    α, x, ∂z_∂state = DCD.proximity_jacobian(cone,capsule)
    α2, x2, ∂z_∂state2 = DCD.proximity_jacobian_slow(cone,capsule)
    α3, x3, ∂z_∂state3 = DCD.proximity_jacobian(capsule,cone)

    @test norm([∂z_∂state3[:,8:14] ∂z_∂state3[:,1:7]] - ∂z_∂state) < 1e-3
    # @btime DCD.proximity($capsule,$cone)
    # @btime DCD.proximity_jacobian($capsule,$cone)

    # check derivatives
    function fd_α(cone,capsule,r1,q1,r2,q2)
        cone.r = r1
        cone.q = q1
        capsule.r = r2
        capsule.q = q2
        α, x = DCD.proximity(cone,capsule; pdip_tol = 1e-6)
        [x;α]
    end

    idx_r1 = SVector{3}(1:3)
    idx_q1 = SVector{4}(4:7)
    idx_r2 = SVector{3}(8:10)
    idx_q2 = SVector{4}(11:14)

    J1 = FiniteDiff.finite_difference_jacobian(θ -> fd_α(cone,capsule,θ[idx_r1],θ[idx_q1],θ[idx_r2],θ[idx_q2]), [cone.r;cone.q;capsule.r;capsule.q])


    @test norm(J1 - ∂z_∂state)  < 1e-2
    @test norm(J1 - ∂z_∂state2) < 1e-2
    @test norm(J1[4,:] - ∂z_∂state[4,:]) < 1e-3
    @test norm(∂z_∂state - ∂z_∂state2) < 1e-5




end
