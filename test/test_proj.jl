#=
This script will test the parallelization of the Dirac kernel with multithreading.
It will implement two things in addition to the serial implementation in LatticeQCD.jl:
add a spinor projection / reconstruction step, and multithread the computation.
=#
using Random
Random.seed!(10)
using LinearAlgebra
using BenchmarkTools
using LatticeQCD.Actions:Actions
using LatticeQCD.WilsonFermion_module:WilsonFermion,Dx!#,mul!
using LatticeQCD.Gaugefields:GaugeFields,RandomGauges,SU3GaugeFields
using LatticeQCD.AbstractFermion:FermionFields#,mul!
# using LatticeQCD.Fermionfields:mul!
include("../src/fermions/ParWilsonFermion.jl")
using ..ParWilsonFermionModule: ParWilsonFermion,fill!,Dx_serial!,project!,mul_dirac!
using HDF5
using Base.Threads
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
println(fill_mat[1, 1, 1, 1, 1, :])
println(ser_ferm.f[1, 1, 1, 1, 1, :])
println(par_ferm.f[1, 1, 1, 1, 1, :])

################################################################################
################################ Test projection ###############################
################################################################################

# function mul!(ferm::FermionFields, A::Matrix{ComplexF64}, x::FermionFields)
#     NX = x.NX
#     NY = x.NY
#     NZ = x.NZ
#     NT = x.NT
#     NC = x.NC
#     for ic=1:NC
#         for it=1:NT
#             for iz=1:NZ
#                 for iy=1:NY
#                     for ix=1:NX
#                         vec = x.f[ic, ix, iy, iz, it, :]
#                         tmp = A * vec
#                         ferm.f[ic, ix, iy, iz, it, :] = tmp
#                     end
#                 end
#             end
#         end
#     end
#     return
# end

# μ = 1
# tmp1_ser = deepcopy(ser_ferm)
# # @btime mul!(tmp1_ser, view(ser_ferm.rminusγ, :, :, μ), tmp1_ser)
# mul!(tmp1_ser, view(ser_ferm.rminusγ, :, :, μ), tmp1_ser)
#
# tmp1_par = deepcopy(par_ferm)
# tmp_halfspinor = ParWilsonFermion(Nc, Nx, Ny, Nz, Nt, 2, ferm_param, bc)
# # @btime project!(tmp1_par, par_ferm, tmp_halfspinor, μ, false)
# project!(tmp1_par, par_ferm, tmp_halfspinor, μ, false)
#
# println("Difference for P^(-$(μ)) = $(maximum(abs.(tmp1_ser.f - tmp1_par.f)))")

t_ser = Array{Float64}(undef, 2, 4)
t_par = Array{Float64}(undef, 2, 4)

println(ser_ferm.rminusγ[:, :, 1])

println("Negative projections")
for μ = 1 : 4
    tmp1_ser = deepcopy(ser_ferm)
    ser_out = WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc)
    mul_dirac!(ser_out, ser_ferm.rminusγ[:, :, μ], tmp1_ser)
    t_ser[1, μ] = @belapsed mul_dirac!($(tmp1_ser), ser_ferm.rminusγ[:, :, $(μ)], $(tmp1_ser))

    par_out = deepcopy(par_ferm)
    tmp1_par = deepcopy(par_ferm)
    tmp_halfspinor = ParWilsonFermion(Nc, Nx, Ny, Nz, Nt, 2, ferm_param, bc)
    project!(par_out, par_ferm, tmp_halfspinor, μ, false)
    t_par[1, μ] = @belapsed project!($(tmp1_par), $(par_ferm), $(tmp_halfspinor), $(μ), false)

    println("Difference for P^(-$(μ)) = $(maximum(abs.(ser_out.f - par_out.f)))")
end
println("Serial projection times: $(t_ser[1, :])")
println("Parallel projection times: $(t_par[1, :])")

println("Positive projections")
for μ = 1 : 4
    tmp1_ser = deepcopy(ser_ferm)
    ser_out = WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc)
    mul_dirac!(ser_out, ser_ferm.rplusγ[:, :, μ], tmp1_ser)
    t_ser[2, μ] = @belapsed mul_dirac!($(tmp1_ser), ser_ferm.rplusγ[:, :, $(μ)], $(tmp1_ser))

    par_out = deepcopy(par_ferm)
    tmp1_par = deepcopy(par_ferm)
    tmp_halfspinor = ParWilsonFermion(Nc, Nx, Ny, Nz, Nt, 2, ferm_param, bc)
    project!(par_out, par_ferm, tmp_halfspinor, μ, true)
    t_par[2, μ] = @belapsed project!($(tmp1_par), $(par_ferm), $(tmp_halfspinor), $(μ), true)

    println("Difference for P^(+$(μ)) = $(maximum(abs.(ser_out.f - par_out.f)))")
end
println("Serial projection times: $(t_ser[2, :])")
println("Parallel projection times: $(t_par[2, :])")

################################################################################
############################ Test projection + shift ###########################
################################################################################

# init random gauge field
# NDW = 1
# U = Array{SU3GaugeFields,1}(undef,4)
# for μ = 1:4
#     U[μ] = RandomGauges(Nc, Nx, Ny, Nz, Nt, NDW)
# end

################################################################################
################################## Save data ###################################
################################################################################
