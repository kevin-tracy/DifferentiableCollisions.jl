# using StaticArrays
# using LinearAlgebra
# using BenchmarkTools
# # import DCD
# # @inline function problem_matrices(prim1::P1,r1::SVector{3,T1},q1::SVector{4,T2},
# #                                   prim2::P2,r2::SVector{3,T3},q2::SVector{4,T4}) where {P1<:AbstractPrimitive,P2<:AbstractPrimitive, T1,T2,T3,T4}
# #
# #
# #     G_ort1, h_ort1, G_soc1, h_soc1 = problem_matrices(prim1,r1,q1)
# #     G_ort2, h_ort2, G_soc2, h_soc2 = problem_matrices(prim2,r2,q2)
# #
# #     combine_problem_matrices(G_ort1,h_ort1,G_soc1, h_soc1, G_ort2,h_ort2, G_soc2, h_soc2)
# # end

@inline function combine_problem_matrices(G_ort1::SMatrix{n_ort1,v1,T1,n_ort1v1},
                                          h_ort1::SVector{n_ort1,T2},
                                          G_soc1::SMatrix{n_soc1,v1,T3,n_soc1v1},
                                          h_soc1::SVector{n_soc1,T4},
                                          G_ort2::SMatrix{n_ort2,v2,T5,n_ort2v2},
                                          h_ort2::SVector{n_ort2,T6},
                                          G_soc2::SMatrix{n_soc2,v2,T7,n_soc2v2},
                                          h_soc2::SVector{n_soc2,T8}) where {n_ort1,n_soc1,n_ort2,n_soc2,v1,v2,n_ort1v1,n_soc1v1,n_ort2v2,n_soc2v2,T1,T2,T3,T4,T5,T6,T7,T8}

    n_ort = n_ort1 + n_ort2
    c = SA[0,0,0,1.0,0,0,0,0,0,0,0,0,0,0,0]
    idx_ort = SVector{n_ort}(1 : (n_ort1 + n_ort2))
    idx_soc1 = SVector{n_soc1}((n_ort + 1):(n_ort + n_soc1))
    idx_soc2 = SVector{n_soc2}((n_ort + n_soc1 + 1):(n_ort + n_soc1 + n_soc2))



    if (v1 == 4) && (v2 == 4) # if they are the same size, just stack
        G = vcat(G_ort1,G_ort2,G_soc1,G_soc2)
        h = vcat(h_ort1,h_ort2,h_soc1,h_soc2)
        return c[SVector{v1}(1:v1)],G,h,idx_ort,idx_soc1,idx_soc2
    elseif (v1 > 4) && (v2 == 4) # if v1 is bigger, add zeros to the second
        G_ort_top = G_ort1
        G_ort_bot = hcat(G_ort2, (@SMatrix zeros(n_ort2,v1-v2)))

        G_soc_top = G_soc1
        G_soc_bot = hcat(G_soc2, (@SMatrix zeros(n_soc2,v1-v2)))

        G = [G_ort_top;G_ort_bot;G_soc_top;G_soc_bot]
        h = [h_ort1;h_ort2;h_soc1;h_soc2]

        return c[SVector{v1}(1:v1)],G,h,idx_ort,idx_soc1,idx_soc2
    elseif (v1 == 4) && (v2 > 4) # if v2 is bigger, add zeros to the first
        G_ort_top = hcat(G_ort1, (@SMatrix zeros(n_ort1,v2-v1)))
        G_ort_bot = G_ort2

        G_soc_top = hcat(G_soc1, (@SMatrix zeros(n_soc1,v2-v1)))
        G_soc_bot = G_soc2

        G = [G_ort_top;G_ort_bot;G_soc_top;G_soc_bot]
        h = [h_ort1;h_ort2;h_soc1;h_soc2]
        return c[SVector{v2}(1:v2)],G,h,idx_ort,idx_soc1,idx_soc2
    elseif (v1 > 4) && (v2 > 4) # if they are both bigger than 4, stack them in the following way:
        # G1 = [G1_p, G1_e]
        # G2 = [G2_p, G2_e]
        #
        # [G1_p, G1_e, 0]
        # [G2_p, 0   , G2_p]

        v1_extra = v1 - 4
        v2_extra = v2 - 4

        if v2_extra == 1
            v2_extra_idx = 5
        else
            v2_extra_idx = SVector{v2_extra}(5:(4 + v2_extra))
        end

        G_ort_top = hcat(G_ort1, (@SMatrix zeros(n_ort1,v2_extra)))
        G_ort_bot = hcat(G_ort2[:,SA[1,2,3,4]], (@SMatrix zeros(n_ort2,v1_extra)), G_ort2[:,v2_extra_idx])

        G_soc_top = hcat(G_soc1, (@SMatrix zeros(n_soc1,v2_extra)))
        G_soc_bot = hcat(G_soc2[:,SA[1,2,3,4]], (@SMatrix zeros(n_soc2,v1_extra)), G_soc2[:,v2_extra_idx])

        G = [G_ort_top;G_ort_bot;G_soc_top;G_soc_bot]
        h = [h_ort1;h_ort2;h_soc1;h_soc2]

        nc = (v1 + v2 - 4)
        return c[SVector{nc}(1:nc)],G,h,idx_ort,idx_soc1,idx_soc2
    end
    error("failure to combine problem matrices")
end

# function stack_two(G1,G2)
#     r1,v1 = size(G1)
#     r2,v2 = size(G2)
#     # @show r1, v1
#     # @show r2, v2
#
#     v1_extra = v1 - 4
#     v2_extra = v2 - 4
#     v1_extra_idx = (v1_extra == 0) ? [] : (5:(v1_extra + 4))
#     v2_extra_idx = (v2_extra == 0) ? [] : (5:(v2_extra + 4))
#
#     # @show v1_extra
#     # @show v2_extra
#     #
#     # @show v1_extra_idx
#     # @show v2_extra_idx
#
#     G1_p = G1[:,1:4]
#     G2_p = G2[:,1:4]
#
#     G1_e = G1[:,v1_extra_idx]
#     G2_e = G2[:,v2_extra_idx]
#
#     [
#     G1_p G1_e               zeros(r1,v2_extra);
#     G2_p zeros(r2,v1_extra) G2_e
#     ]
# end
# let
#     # cone = DCD.Cone(2.0,deg2rad(22))
#     # cone.r = 0.3*(@SVector randn(3))
#     # cone.q = normalize((@SVector randn(4)))
#     #
#     # capsule = DCD.Capsule(.3,1.2)
#     # capsule.r = (@SVector randn(3))
#     # capsule.q = normalize((@SVector randn(4)))
#     #
#     # G,h,idx_ort,idx_soc1,idx_soc2 = problem_matrices(cone,cone.r,cone.q,capsule,capsule.r,capsule.q)
#     # @btime problem_matrices($cone,$cone.r,$cone.q,$capsule,$capsule.r,$capsule.q)
#     n_ort1 = 2
#     n_ort2 = 5
#     n_soc1 = 3
#     n_soc2 = 4
#
#     v1s = [6,4,4,4,7,6]
#     v2s = [6,4,5,5,6,7]
#     for i = 1:length(v1s)
#
#         v1 = v1s[i]
#         v2 = v2s[i]
#
#         G_ort1 = @SMatrix randn(n_ort1,v1)
#         h_ort1 = @SVector randn(n_ort1)
#         G_ort2 = @SMatrix randn(n_ort2,v2)
#         h_ort2 = @SVector randn(n_ort2)
#
#         G_soc1 = @SMatrix randn(n_soc1,v1)
#         h_soc1 = @SVector randn(n_soc1)
#         G_soc2 = @SMatrix randn(n_soc2,v2)
#         h_soc2 = @SVector randn(n_soc2)
#
#         c,G,h,idx_ort,idx_soc1,idx_soc2 = combine_problem_matrices(G_ort1,h_ort1,G_soc1, h_soc1, G_ort2,h_ort2, G_soc2, h_soc2)
#         @btime combine_problem_matrices($G_ort1,$h_ort1,$G_soc1, $h_soc1, $G_ort2,$h_ort2, $G_soc2, $h_soc2)
#         G2 = [stack_two(G_ort1,G_ort2);stack_two(G_soc1,G_soc2)]
#         h2 = [h_ort1;h_ort2;h_soc1;h_soc2]
#         @info norm(G-G2), norm(h - h2)
#
#     end
#     #
#     # v1 = 6
#     # v2 = 4
#     #
#     # G_ort1 = @SMatrix randn(n_ort1,v1)
#     # h_ort1 = @SVector randn(n_ort1)
#     # G_ort2 = @SMatrix randn(n_ort2,v2)
#     # h_ort2 = @SVector randn(n_ort2)
#     #
#     # G_soc1 = @SMatrix randn(n_soc1,v1)
#     # h_soc1 = @SVector randn(n_soc1)
#     # G_soc2 = @SMatrix randn(n_soc2,v2)
#     # h_soc2 = @SVector randn(n_soc2)
#     #
#     # G,h,idx_ort,idx_soc1,idx_soc2 = combine_problem_matrices(G_ort1,h_ort1,G_soc1, h_soc1, G_ort2,h_ort2, G_soc2, h_soc2)
#     # @btime combine_problem_matrices($G_ort1,$h_ort1,$G_soc1, $h_soc1, $G_ort2,$h_ort2, $G_soc2, $h_soc2)
#     #
#     # v1 = 4
#     # v2 = 6
#     #
#     # G_ort1 = @SMatrix randn(n_ort1,v1)
#     # h_ort1 = @SVector randn(n_ort1)
#     # G_ort2 = @SMatrix randn(n_ort2,v2)
#     # h_ort2 = @SVector randn(n_ort2)
#     #
#     # G_soc1 = @SMatrix randn(n_soc1,v1)
#     # h_soc1 = @SVector randn(n_soc1)
#     # G_soc2 = @SMatrix randn(n_soc2,v2)
#     # h_soc2 = @SVector randn(n_soc2)
#     #
#     # G,h,idx_ort,idx_soc1,idx_soc2 = combine_problem_matrices(G_ort1,h_ort1,G_soc1, h_soc1, G_ort2,h_ort2, G_soc2, h_soc2)
#     # @btime combine_problem_matrices($G_ort1,$h_ort1,$G_soc1, $h_soc1, $G_ort2,$h_ort2, $G_soc2, $h_soc2)
#     #
#     # v1 = 5
#     # v2 = 5
#     #
#     # G_ort1 = @SMatrix randn(n_ort1,v1)
#     # h_ort1 = @SVector randn(n_ort1)
#     # G_ort2 = @SMatrix randn(n_ort2,v2)
#     # h_ort2 = @SVector randn(n_ort2)
#     #
#     # G_soc1 = @SMatrix randn(n_soc1,v1)
#     # h_soc1 = @SVector randn(n_soc1)
#     # G_soc2 = @SMatrix randn(n_soc2,v2)
#     # h_soc2 = @SVector randn(n_soc2)
#     #
#     # G,h,idx_ort,idx_soc1,idx_soc2 = combine_problem_matrices(G_ort1,h_ort1,G_soc1, h_soc1, G_ort2,h_ort2, G_soc2, h_soc2)
#     # @btime combine_problem_matrices($G_ort1,$h_ort1,$G_soc1, $h_soc1, $G_ort2,$h_ort2, $G_soc2, $h_soc2)
#
# end
