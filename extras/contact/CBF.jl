using Pkg
Pkg.activate(dirname(@__DIR__))
using LinearAlgebra
using StaticArrays
import ForwardDiff as fd
import FiniteDiff as FD2
using Printf
# using SparseArrays
import MeshCat as mc
import DCOL as dc
using JLD2
# using MATLAB
# import DifferentialProximity as dp
import Random
using Convex
using ECOS
using Colors

function ϕ(x,Pvic,P1)
    Pvic.r = SA[x[1],x[2],x[3]]
    # xc = SA[4,4]
    # -(norm(xc - x[1:2])^2 - 2^2) #>= 0
    -(dc.proximity(Pvic,P1)[1] - 1)
end
# function ϕ2(x)
#     xc = SA[7,7]
#     -(norm(xc - x[1:2])^2 - 1^2) #>= 0
# end
function ∇ϕ(x,Pvic,P1)
    # FD2.finite_difference_gradient(_x -> ϕ(_x,Pvic,P1),x)
    Pvic.r = SA[x[1],x[2],x[3]]
    _,_,J = dc.proximity_jacobian(Pvic,P1)
    [-J[4,1:3];zeros(3)]
end
# function ∇ϕ2(x)
#     FD.gradient(ϕ2,x)
# end
function ϕ̇(x,u,Ad,Bd,dt,Pvic,P1)
    x2 = Ad*x + Bd*u
    xdot = (x2 - x)/dt
    ∇ϕ(x,Pvic,P1)'xdot
end

function CBF_convex(x,u_nom,Ad,Bd,dt,Pvic,P1)
    λ = -.5
    u = Variable(3)
    x2 = Ad*x + Bd*u
    xdot = (x2 - x)/dt
    p = minimize(norm(u_nom - u))
    p.constraints += ∇ϕ(x,Pvic,P1)'*xdot  <= λ*ϕ(x,Pvic,P1)
    # p.constraints += ∇ϕ2(x)'*xdot <= λ*ϕ2(x)
    p.constraints += norm(u,Inf)<=4
    solve!(p,()->ECOS.Optimizer())
    return vec(u.value)
end
function vis_traj!(vis, name, X; R = 0.1, color = mc.RGBA(1.0, 0.0, 0.0, 1.0))
    for i = 1:(length(X)-1)
        a = X[i][1:3]
        b = X[i+1][1:3]
        cyl = mc.Cylinder(mc.Point(a...), mc.Point(b...), R)
        mc.setobject!(vis[name]["p"*string(i)], cyl, mc.MeshPhongMaterial(color=color))
    end
    for i = 1:length(X)
        a = X[i][1:3]
        sph_p1 = mc.HyperSphere(mc.Point(a...), R)
        mc.setobject!(vis[name]["s"*string(i)], sph_p1, mc.MeshPhongMaterial(color = color))
    end
end
let

    nx = 6
    nu = 3
    A = Matrix([zeros(3,3) I(3); zeros(3,6)])
    B = Matrix([zeros(3,3); I(3)])
    dt = .2
    H = exp(dt*[A B; zeros(nu,nx + nu)])
    Ad = (H[1:nx,1:nx])
    Bd = (H[1:nx,(nx+1):end])

    N = 100

    X = [zeros(nx) for i = 1:N]
    U = [zeros(nu) for i = 1:N-1]
    phi = zeros(N-1)
    phidot = zeros(N-1)

    P1 = dc.Cone(3.0,deg2rad(22))
    P1.r = SA[0,0,0.8]
    using Random
    Random.seed!(5)
    P1.q = normalize((@SVector randn(4)))
    # P2 = dc.Sphere(1)
    # P2.r = SA[7,7,0.0]
    # Pvic = dc.Sphere(0.5)
    Pvic = dc.Cylinder(0.5,1.0)

    X[1] = [8,5,1.0,0.1,0.1,0]
    xg = [-4,-4,0]
    kp = .10
    kd = .5
    for i = 1:N-1
        U[i] = -kp*(X[i][1:3]-xg) - kd*X[i][4:6]
        U[i] = CBF_convex(X[i],U[i],Ad,Bd,dt,Pvic,P1)
#         U[i] = CBF_tinyqp(X[i],U[i],Ad,Bd,dt)
        X[i+1] = Ad*X[i] + Bd*U[i]
        phi[i] = ϕ(X[i],Pvic,P1)
        # phidot[i] = ϕ̇(X[i],U[i],Ad,Bd,dt)
    end

    vis = mc.Visualizer()
    mc.open(vis)

    c1 = [245, 155, 66]/255
    c2 = [2,190,207]/255

    dc.build_primitive!(vis, P1, :P1; α = 1.0,color = mc.RGBA(c1..., 1.0))
    # dc.build_primitive!(vis, P2, :P2; α = 1.0,color = mc.RGBA(normalize(abs.(randn(3)))..., 1.0))
    dc.build_primitive!(vis, Pvic, :Pvic; α = 1.0,color = mc.RGBA(c2..., 1.0))
    dc.update_pose!(vis[:P1],P1)
    # dc.update_pose!(vis[:P2],P2)


    # for i = 1:length(P_obs)
        # sph_p1 = mc.HyperSphere(mc.Point(0,0,0.0), 0.3)
        # mc.setobject!(vis[:vic], sph_p1, mc.MeshPhongMaterial(color = mc.RGBA(1.0,0,0,1.0)))
    # end
    mc.setprop!(vis["/Lights/AmbientLight/<object>"], "intensity", 0.9)
    mc.setprop!(vis["/Background"], "top_color", colorant"transparent")
    mc.setvisible!(vis["/Grid"],false)
    dc.set_floor!(vis; x = 20, y = 20)
    vis_traj!(vis, :traj, X; R = 0.1, color = mc.RGBA(1.0, 0.0, 0.0, 1.0))

    anim = mc.Animation(floor(Int,1/dt))
    # for k = 1:N
    #     mc.atframe(anim, k) do
    #         mc.settransform!(vis[:Pvic],mc.Translation(X[k][1:3]))
    #     end
    # end
    # mc.setanimation!(vis, anim)
    for k = 1:N
        mc.atframe(anim, k) do
             # mc.settransform!(vis["/Cameras/default"], mc.Translation(cam_ps[k]))
            # for i = 1:length(P_obs)
                # name = "P" * string(i)
                mc.settransform!(vis[:P1], mc.Translation(P1.r - X[k][1:3]) ∘ mc.LinearMap(dc.dcm_from_q(P1.q)))

            # end
            mc.settransform!(vis[:floor], mc.Translation(-X[k][1:3]))
            mc.settransform!(vis[:traj], mc.Translation(-X[k][1:3]))
        end
    end
    mc.setanimation!(vis, anim)


    # Xm = Matrix(hcat(X...))
    # mat"
    # figure
    # hold on
    # plot($Xm(1,:),$Xm(2,:),'b-o')
    # p = nsidedpoly(1000, 'Center', [4 4], 'Radius', 2);
    # plot(p, 'FaceColor', 'r')
    # p = nsidedpoly(1000, 'Center', [7 7], 'Radius', 1);
    # plot(p, 'FaceColor', 'r')
    # axis equal
    # hold off
    # "
#     mat"
#     figure
#     hold on
#     plot($phi)
#     hold off
#     "
#     mat"
#     figure
#     hold on
#     plot($phidot)
#     plot(-1*$phi,'r')
#     hold off
#     "
#     mat"
#     figure
#     hold on
#     plot($Xm')
#     hold off
#     "

end
