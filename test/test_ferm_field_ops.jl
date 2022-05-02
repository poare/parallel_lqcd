#=
This script will test the parallelization of the Dirac kernel with multithreading.
It will implement two things in addition to the serial implementation in LatticeQCD.jl:
add a spinor projection / reconstruction step, and multithread the computation.
=#

using BenchmarkTools
using LatticeQCD.Actions:Actions
using LatticeQCD.WilsonFermion_module:WilsonFermion
using LatticeQCD.AbstractFermion:FermionFields,set_wing_fermi!,gauss_distribution_fermi!
include("../src/fermions/ParWilsonFermion.jl")
using ..ParWilsonFermionModule: ParWilsonFermion,fill!,set_wing_fermi_threads!
using Base.Threads
using HDF5
n_threads = Threads.nthreads()
println("Using $(n_threads) thread(s).")

# Set up a gauge action
mq = -0.2450
κ = 1 / (2 * mq + 8)                            # hopping parameter
ferm_param = Actions.FermiActionParam_Wilson(κ, 1, 1e-16, 3000)
Nc = 3
# Nx = 8; Ny = 8; Nz = 8; Nt = 8
Nx = 16; Ny = 16; Nz = 16; Nt = 48
bc = ones(Int8, 4)

# initialize fermions
ser_ferm = WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc)
fill_mat = rand(eltype(ser_ferm.f), size(ser_ferm.f))
fill!(ser_ferm, fill_mat)
# gauss_distribution_fermi!(ser_ferm)
par_ferm = ParWilsonFermion(ser_ferm)

################################################################################
################################ Test wingfermi! ###############################
################################################################################

# Test __rmul__
println("Testing wingfermi! overloading.")
@btime set_wing_fermi!(ser_ferm)
@btime set_wing_fermi_threads!(par_ferm)
println("Maximum deviation between parallel and serial wingfermi! is $(maximum(abs.(ser_ferm.f - par_ferm.f)))")

#=
Multithreading wing_fermi! actually makes it slower! The strange loop structure that the original wing_fermi! has is likely optimized
by Julia's compiler. We should only use the original wing_fermi!, not this multithreaded version.
=#
