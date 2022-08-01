
function mass_properties(cone::Union{Cone{T},ConeMRP{T}}; ρ = 1.0) where {T}
    # https://en.wikipedia.org/wiki/List_of_moments_of_inertia
    r = tan(cone.β)*cone.H
    V = (1/3)*(pi*(r^2)*cone.H)
    m = V*ρ

    Iyy = m*((3/20)*r^2 + (3/80)*cone.H^2)
    Izz = Iyy
    Ixx = 0.3*m*r^2

    return m, Diagonal(SA[Ixx,Iyy,Izz])
end

function mass_properties(capsule::Union{Capsule{T},CapsuleMRP{T}}; ρ = 1.0) where {T}
    # https://www.gamedev.net/tutorials/programming/math-and-physics/capsule-inertia-tensor-r3856/
    R = capsule.R
    L = capsule.L
    V_cyl = pi*(capsule.R^2)*capsule.L
    V_hs = (2*pi*(capsule.R^3)/3)
    m = (V_cyl + 2*V_hs)*ρ

    m_cyl = V_cyl * ρ
    m_hs = V_hs * ρ

    Ixx = m_cyl*R^2/2 + 2*m_hs*(2*R^2/5)
    Iyy = m_cyl * (L^2/12 + R^2/4) + 2 * m_hs * (2*R^2/5 + L^2/2 + 3*L*R/8)
    Izz = m_cyl * (L^2/12 + R^2/4) + 2 * m_hs * (2*R^2/5 + L^2/2 + 3*L*R/8)

    return m, Diagonal(SA[Ixx,Iyy,Izz])
end
