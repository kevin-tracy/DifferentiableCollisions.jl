import Pkg; Pkg.activate("/Users/kevintracy/.julia/dev/DCD/extras")
using LinearAlgebra
using StaticArrays
import ForwardDiff as fd
using JLD2

@load "/Users/kevintracy/.julia/dev/DCD/extras/example_socp.jld2"


function c_no(x)
    G_ort*x - h_ort # in negative orthant
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
function AL(x,μ,λ)
    cost(x) + (1/(2*μ))*(norm(Π(λ - μ*con(x)))^2 - dot(λ,λ))
end
function linesearch(x,Δx,μ,λ)
    J1 = AL(x,μ,λ)
    α = 1.0
    for i = 1:20
        J2 = AL(x + α*Δx, μ,λ)
        if J2 < J1
            return α
        else
            α *= 0.5
        end
    end
    error("linesearch failed")
end
let

    # solve AL
    n_ort = length(h_ort)
    n_soc = length(h_soc)
    λ = zeros(n_ort + n_soc)
    x = randn(5)
    μ = 1.0
    ϕ = 10.0

    for i = 1:20

        # newton steps for AL minimization
        H = fd.hessian(_x -> AL(_x,μ,λ), x)
        g = fd.gradient(_x -> AL(_x,μ,λ), x)
        Δx = -(H + 0e-6*I)\g
        α = linesearch(x,Δx,μ,λ)
        x += α*Δx

        @show norm(g), α, norm(con(x) - Π(con(x)))

        if norm(con(x) - Π(con(x))) < 1e-3
            @info "success"
            break
        end
        # AL min has converged, time to update
        if norm(Δx)<1e-3
            @info "penalty update"
            λ .= Π(λ - μ*con(x))
            μ *= ϕ
        end

    end

end
