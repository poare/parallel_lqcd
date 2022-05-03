#=
This script will test the parallelization of the Dirac kernel with multithreading.
It will implement two things in addition to the serial implementation in LatticeQCD.jl:
add a spinor projection / reconstruction step, and multithread the computation.
=#

using BenchmarkTools
using LatticeQCD.Actions:Actions
using LatticeQCD.WilsonFermion_module:WilsonFermion,Dx!
using LatticeQCD.Gaugefields:GaugeFields,RandomGauges,SU3GaugeFields
include("../src/fermions/ParWilsonFermion.jl")
include("../src/solvers/ConjGradientSolver.jl")
using ..ParWilsonFermionModule: ParWilsonFermion,fill_scalar!,Dx_serial!,fill!
using ..ConjGradientSolverModule: solve_verbose_serial_Dx!
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
Nx = 4; Ny = 4; Nz = 4; Nt = 8
bc = ones(Int8, 4)

# initialize fermions
ser_ferm = WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc)
fill_mat = rand(eltype(ser_ferm.f), size(ser_ferm.f))
fill!(ser_ferm, fill_mat)
par_ferm = ParWilsonFermion(ser_ferm)

# init random gauge field
NDW = 1
U = Array{SU3GaugeFields,1}(undef,4)
for μ = 1:4
    U[μ] = RandomGauges(Nc, Nx, Ny, Nz, Nt, NDW)
end

################################################################################
############################### Init and run test ##############################
################################################################################

par_ferm_out = ParWilsonFermion(ser_ferm)

temp_ser = [
    WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc),
    WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc),
    WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc),
    WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc)
    ]
temp_par = [ParWilsonFermion(ferm) for ferm = temp_ser]

fill_scalar!(par_ferm_out,0)
fill_scalar!(par_ferm,0)
par_ferm[1,1,1,1,1,1]=1.
par_ferm[2,1,1,1,1,1]=0+1im

workspace_ferm=par_ferm_out+par_ferm

Dx_serial!(workspace_ferm, U, par_ferm, temp_par)


solve_verbose_serial_Dx!(par_ferm_out,U, temp_par,par_ferm)

test_zero=par_ferm - workspace_ferm
test_error=sqrt(test_zero*test_zero)
println("Norm of residual difference of Dx-b: ", sqrt(test_zero*test_zero))
