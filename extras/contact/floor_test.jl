


function create_rect_prism(;len = 20.0, wid = 20.0, hei = 2.0)
# len = 20.0
# wid = 20.0
# hei = 2.0

    ns = [SA[1,0,0.0], SA[0,1,0.0], SA[0,0,1.0],SA[-1,0,0.0], SA[0,-1,0.0], SA[0,0,-1.0]]
    cs = [SA[len,0,0.0], SA[0,wid,0.0], SA[0,0,hei],SA[-len,0,0.0], SA[0,-wid,0.0], SA[0,0,-hei]]

    A = zeros(6,3)
    b = zeros(6)

    for i = 1:6
        A[i,:] = ns[i]'
        b[i] = dot(ns[i],cs[i])
    end

    A = SMatrix{6,3}(A)
    b = SVector{6}(b)

    mass = len*wid*hei

    inertia = (mass/12)*Diagonal(SA[wid^2 + hei^2, len^2 + hei^2, len^2 + wid^2])

    return dc.Polytope(A,b), mass, inertia
end

# vis = mc.Visualizer()
# mc.open(vis)
# P = dc.Polytope(A,b)
# dc.build_primitive!(vis, P, :test ; α = 1.0,color = mc.RGBA(normalize(abs.(randn(3)))..., 1.0))
#
# for i = 1:N_bodies
#     dc.build_primitive!(vis, P[i], Symbol("P"*string(i)); α = 1.0,color = mc.RGBA(normalize(abs.(randn(3)))..., 1.0))
# end
