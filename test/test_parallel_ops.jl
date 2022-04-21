using BenchmarkTools
# using LinearAlgebra
using LatticeQCD.Actions:Actions
using LatticeQCD.WilsonFermion_module:WilsonFermion,LinearAlgebra.rmul!,Wx!
# using ..src/fermions/ParWilsonFermion            # TODO figure this out
# using ParWilsonFermion
include("../src/fermions/ParWilsonFermion.jl")
using ..ParWilsonFermionModule: ParWilsonFermion
# using .ParWilsonFermionModule
print("Loaded ParWilsonFermionModule")

# Set up a gauge action
mq = -0.2450
κ = 1 / (2 * mq + 8)                            # hopping parameter
ferm_param = Actions.FermiActionParam_Wilson(κ, 1, 1e-16, 3000)
Actions.show_parameters_action(ferm_param)
Nc = 3
Nx = 8; Ny = 8; Nz = 8; Nt = 8
bc = ones(Int8, 1)
mul_factor = 4.3

# enable multithreading: TODO

# initialize fermions
ser_ferm = WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc)
par_ferm = ParWilsonFermion(ser_ferm)                           # can construct either way
# par_ferm = ParWilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc)

# Test __rmul__
println("Testing __rmul__ overloading.")
@btime rmul!(ser_ferm, mul_factor)
@btime rmul!(par_ferm, mul_factor)

# Test __operation___ TODO
