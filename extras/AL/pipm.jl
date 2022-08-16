import Pkg; Pkg.activate("/Users/kevintracy/.julia/dev/DCD/extras")
using LinearAlgebra
using StaticArrays
import ForwardDiff as fd
using JLD2

@load "/Users/kevintracy/.julia/dev/DCD/extras/example_socp.jld2"


function c_po(x)
    h_ort - G_ort*x # in negative orthant
end
function bar_po(u)
    -sum(log.(u))
end
function bar_soc(u)
    us = u[1]
    uv = u[2:end]
    -0.5*log(us^2 - dot(uv,uv))
end
function ϕ(x)
    bar_po(c_po(x)) + bar_soc(c_soc(x))
end
function c_soc(x)
    h_soc - G_soc*x # in SOC
end
function con(x)
    [
    c_no(x);
    c_soc(x)
    ]
end
function Π_no(x)
    min.(0,x)
end
function Π_soc(x)
    @assert length(x) == length(h_soc)
    s = x[1]
    v = x[2:end]
    if norm(v) <= -s
        return 0*x
    elseif norm(v) <= s
        return x
    elseif norm(v) >= abs(s)
        return 0.5*(1 + s/norm(v))*[norm(v);v]
    end
    error("nothing happened on projection")
end
function Π(x)
    n_ort = length(h_ort)
    n_soc = length(h_soc)
    [
    Π_no(x[1:n_ort]);
    Π_soc(x[(n_ort + 1):(n_ort + n_soc)])
    ]
end
function cost(x)
    x[4]
end
function ipm_cost(x, t)
    cost(x) + (1/t)*ϕ(x)
end
function is_feasible(x)
    s_ort = h_ort - G_ort*x
    s_soc = h_soc - G_soc*x
    if (minimum(s_ort)>0) && (norm(s_soc[2:end]) < s_soc[1])
        return true
    else
        return false
    end
end

function linesearch(x,Δx,t)
    J1 = ipm_cost(x,t)
    α = 1.0
    for i = 1:20
        x2 = x + α*Δx
        if is_feasible(x2)
            J2 = ipm_cost(x2, t)
            if J2 < J1
                return α
            else
                α *= 0.5
            end
        else
            α *= 0.5
        end
    end
    error("linesearch failed")
end
let

    # solve IPM
    x = [0,0,0,1e1,0]
    t = 1e0

    s_ort = h_ort - G_ort*x

    @show minimum(s_ort)

    s_soc = h_soc - G_soc*x

    @show s_soc[1]
    @show norm(s_soc[2:end])

    @show ϕ(x)

    for i = 1:30
    #
    #     # newton steps for AL minimization
        H = fd.hessian(_x -> ipm_cost(_x,t), x)
        g = fd.gradient(_x -> ipm_cost(_x,t), x)
        if norm(g)<1e-6
            @info "penalty update"
            t *= 10
        else
            Δx = -(H + 0e-6*I)\g
            α = linesearch(x,Δx,t)
            x += α*Δx
        #
            @show norm(g), α, norm(con(x) - Π(con(x))), t
        end
    #
    #     # if norm(con(x) - Π(con(x))) < 1e-3
    #     #     @info "success"
    #     #     break
    #     # end
    #     # # AL min has converged, time to update
    #     # if norm(Δx)<1e-3
    #     #     @info "penalty update"
    #     #     λ .= Π(λ - μ*con(x))
    #     #     μ *= ϕ
    #     # end
    #
    end

end
