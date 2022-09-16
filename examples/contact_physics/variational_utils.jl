
function hat(ω)
    return [0 -ω[3] ω[2];
            ω[3] 0 -ω[1];
            -ω[2] ω[1] 0]
end
function L(Q)
    [Q[1] -Q[2:4]'; Q[2:4] Q[1]*I + hat(Q[2:4])]
end

function R(Q)
    [Q[1] -Q[2:4]'; Q[2:4] Q[1]*I - hat(Q[2:4])]
end

function ρ(ϕ)
    (1/(sqrt(1 + dot(ϕ,ϕ))))*[1;ϕ]
end

const H = [zeros(1,3); I];

const gravity = [0;0;-9.81]

const T = Diagonal([1.0; -1; -1; -1])

function G(Q)
    return L(Q)*H
end
function Expq(ϕ)
    # The quaternion exponential map ϕ → q
    θ = norm(ϕ)
    q = [cos(θ/2); 0.5*ϕ*sinc(θ/(2*pi))]
end
