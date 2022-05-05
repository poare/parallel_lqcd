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
using ..ParWilsonFermionModule: ParWilsonFermion,fill!,Dx_serial!,Dx_halfspinor!
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
par_ferm = ParWilsonFermion(ser_ferm)

# init random gauge field
NDW = 1
U = Array{SU3GaugeFields,1}(undef,4)
for μ = 1:4
    U[μ] = RandomGauges(Nc, Nx, Ny, Nz, Nt, NDW)
end

################################################################################
################################ Dx! halfspinor ################################
################################################################################

ser_ferm_out = WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc)
par_ferm_out = ParWilsonFermion(ser_ferm_out)
par_ferm_halfspinor_out = ParWilsonFermion(ser_ferm_out)
temp_ser = [
    WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc),
    WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc),
    WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc),
    WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc),
    WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc)
    ]
temp_par = [ParWilsonFermion(ferm) for ferm = temp_ser]
full_temps = [
    ParWilsonFermion(Nc, Nx, Ny, Nz, Nt, 4, ferm_param, bc),
    ParWilsonFermion(Nc, Nx, Ny, Nz, Nt, 4, ferm_param, bc),
    ParWilsonFermion(Nc, Nx, Ny, Nz, Nt, 4, ferm_param, bc)
]
half_temps = [
    ParWilsonFermion(Nc, Nx, Ny, Nz, Nt, 2, ferm_param, bc),
    ParWilsonFermion(Nc, Nx, Ny, Nz, Nt, 2, ferm_param, bc),
    ParWilsonFermion(Nc, Nx, Ny, Nz, Nt, 2, ferm_param, bc),
    ParWilsonFermion(Nc, Nx, Ny, Nz, Nt, 2, ferm_param, bc)
]

# Dx_serial!(par_ferm_out, U, par_ferm, temp_par)
# Dx_halfspinor!(par_ferm_halfspinor_out, U, par_ferm, full_temps, half_temps)
t_ser = @belapsed Dx_serial!(par_ferm_out, U, par_ferm, temp_par)
t_half = @belapsed Dx_halfspinor!(par_ferm_halfspinor_out, U, par_ferm, full_temps, half_temps)

# compare timings and output fermions
println("Standard Dx! time: $(t_ser)")
println("Halfspinor Dx! time: $(t_half)")
println("Maximum deviation between regular and halfspinor Dx! output is $(maximum(abs.(par_ferm_out.f - par_ferm_halfspinor_out.f)))")

################################################################################
################################## Save data ###################################
################################################################################

# if n_threads == 1
#     cmd = "w"                   # create new file
# else
#     cmd = "cw"                  # open old file
# end
# fout = h5open("/Users/theoares/parallel_lqcd/data/", cmd)
# times = [t0, t1, t2, t3]
# # times = [t0, t1, t2, t3, t4]
# fout["nt$(n_threads)"] = times
# close(fout)
