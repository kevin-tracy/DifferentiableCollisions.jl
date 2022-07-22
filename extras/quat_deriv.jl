using LinearAlgebra
import ForwardDiff as fd
using Test
using FiniteDiff

function dcm_from_q(q)
    q4,q1,q2,q3 = normalize(q)
    # q = q/norm(q)
    # q4,q1,q2,q3 = q

    # DCM
    [(2*q1^2+2*q4^2-1)   2*(q1*q2 - q3*q4)   2*(q1*q3 + q2*q4);
          2*(q1*q2 + q3*q4)  (2*q2^2+2*q4^2-1)   2*(q2*q3 - q1*q4);
          2*(q1*q3 - q2*q4)   2*(q2*q3 + q1*q4)  (2*q3^2+2*q4^2-1)]
end
function dcm_from_q2(q)
    q = q/norm(q)
    s = q[1]
    v = q[2:4]
    I + 2*hat(v)*(s*I + hat(v))
end

function f1(q,a)
    dcm_from_q(q)*a
end
function f2(q,a)
    a + 2*q[1]*cross(q[2:4],a) + 2*cross(q[2:4],cross(q[2:4],a))
end
function f3(q,a)
    dcm_from_q2(q)*a
end
function hat(p)
    # cross product matrix
    [0 -p[3] p[2]; p[3] 0 -p[1]; -p[2] p[1] 0]
end

let


    # for i = 1:1
    #     q = normalize(randn(4))
    #     aa = randn(3)
    #
    #     Q1 = dcm_from_q(q)
    #     Q2 = dcm_from_q2(q)
    #     @test norm(Q1-Q2) < 1e-13
    #
    #     @test norm(f1(q,aa) - f2(q,aa)) < 1e-13
    #     @test norm(f1(q,aa) - f3(q,aa)) < 1e-13
    #
    #     J1 = fd.jacobian(_q -> f1(_q,aa), q)
    #     J2 = fd.jacobian(_q -> f2(_q,aa), q)
    #     J3 = fd.jacobian(_q -> f3(_q,aa), q)
    #
    #     @info "start"
    #     @show J1
    #     @show J2
    #     @show J3
    #     # @test norm(J1-J2) < 1e-13
    #     # @test norm(J1-J3) < 1e-13
    #     # @test norm(J2 - J3) < 1e-13
    #     # @info "done"
    # end

    # for i = 1:100

    q = normalize(randn(4))
    s = q[1]
    v = q[2:4]
    a = randn(3)

    Q1 = dcm_from_q(q)
    Q2 = I + 2*hat(v)*(s*I + hat(v))

    @test norm(Q1-Q2) < 1e-13

    # J1 = fd.jacobian(_q -> dcm_from_q2(_q)*a, q)

    x2 = Q1*a
    x22 = a + 2*s*cross(v,a) + 2*cross(v,cross(v,a))
    @test norm(x2 - x22) < 1e-13

    # chain rule for dcm(normalize(q)), we will say q̄ = normalize(q)
    #d_dcm/d_normalize(q)*d_normalize(q)/dq

    # first is ∂dcm_∂q̄
    ∂dcm_∂q̄ = [2*cross(v,a)  -2*s*hat(a)] + [zeros(3) 2*(v*a' + (v'*a)*I - 2*a*v')]

    # now we do ∂(normalize(q))_∂q
    t0 = norm(q) # this should be 1 always for quaternions
    ∂q̄_∂q = (1/t0)*I - (1/t0^3)*q*q'

    # analytical jacobian
    J_analytical = ∂dcm_∂q̄*∂q̄_∂q

    # check with forward diff
    J_fd1 =  fd.jacobian(_q -> dcm_from_q(_q)*a, q)
    J_fd2 =  fd.jacobian(_q -> dcm_from_q2(_q)*a, q)
    @test norm(J_fd1 - J_fd2) < 1e-13

    @show J_real1
    @show J_analytical
    @show norm(J_real1 - J_analytical)

    # J1 = fd.jacobian(_q -> dcm_from_q(_q)*a, q)
    # J2 =
    # # vv = 2*cross(v,cross(v,a))
    # # vv2 = 2*((v'*a)*v - (v'*v)*a)
    # # @show norm(vv - vv2)
    #
    # J1 = fd.jacobian(_v -> 2*cross(_v,cross(_v,a)), v )
    # J2 = 2*(v*a' + (v'*a)*I - 2*a*v')
    # @test norm(J1-J2) < 1e-13
    #
    # J1 = fd.jacobian(_q -> 2*cross(_q[2:4],cross(_q[2:4],a)), q )
    # J2 = [zeros(3) 2*(v*a' + (v'*a)*I - 2*a*v')]
    # @test norm(J1-J2) < 1e-13
    #
    # J1 = fd.jacobian(_q -> 2*_q[1]*cross(_q[2:4],a),q)
    # J2 = [2*cross(v,a)  -2*s*hat(a)]
    # @test norm(J1-J2) < 1e-13
    #
    # J1 = fd.jacobian(_q -> a + 2*_q[1]*cross(_q[2:4],a) + 2*cross(_q[2:4],cross(_q[2:4],a)), q  )
    # J2 = fd.jacobian(_q -> dcm_from_q(_q)*a, q)
    # # @test norm(J1 - J2) < 1e-13
    #
    # x1 = a + 2*q[1]*cross(q[2:4],a) + 2*cross(q[2:4],cross(q[2:4],a))
    # x2 = dcm_from_q(q)*a
    # @test norm(x1-x2) < 1e-13
    #
    # # J1 = fd.jacobian(_q -> f1(_q,a), q)
    # # J2 = fd.jacobian(_q -> f2(_q,a), q)
    # #
    # # # @show f1(q,a)
    # # # @show f2(q,a)
    # # @show J1
    # # @show J2
    #
    # # for i = 1:10000
    # #     q = normalize(randn(4))
    # #     a = randn(3)
    # #     @test norm(f1(q,a) - f2(q,a)) < 1e-13
    # # end
    #
    # # J1 = FiniteDiff.finite_difference_jacobian(_q -> f1(_q,a), q)
    # # J2 = FiniteDiff.finite_difference_jacobian(_q -> f2(_q,a), q)
    # # @show J1
    # # @show J2
    # # @test norm(J1-J2) < 1e-13

end
