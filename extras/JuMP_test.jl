using JuMP
using JLD2
using COSMO
using ECOS
using Hypatia
using BenchmarkTools

@load "example_socp.jld2"


# model = JuMP.Model(optimizer_with_attributes(COSMO.Optimizer, "verbose" => true));
# model = JuMP.Model(ECOS.Optimizer)
model = JuMP.Model(Hypatia.Optimizer)
set_optimizer_attribute(model, "verbose", 0)
@variable(model, x[1:5])
@objective(model, Min, x[4])
@constraint(model,c1, G_ort*x .<= h_ort)
@constraint(model,c2, h_soc - G_soc*x in SecondOrderCone())
status = JuMP.optimize!(model)
# @benchmark status = JuMP.optimize!($model)

@show value.(x)
@show dual.(c1)
@show dual.(c2)
