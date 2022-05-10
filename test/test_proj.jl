#=
This script will test the parallelization of the Dirac kernel with multithreading.
It will implement two things in addition to the serial implementation in LatticeQCD.jl:
add a spinor projection / reconstruction step, and multithread the computation.
=#
using Random
Random.seed!(10)
using LinearAlgebra
using BenchmarkTools
using StaticArrays
using LatticeQCD.Actions:Actions
using LatticeQCD.WilsonFermion_module:WilsonFermion,Dx!,fermion_shift!
using LatticeQCD.Gaugefields:GaugeFields,RandomGauges,SU3GaugeFields
using LatticeQCD.AbstractFermion:FermionFields#,set_wing_fermi!
# using LatticeQCD.Fermionfields:mul!
include("../src/fermions/ParWilsonFermion.jl")
using ..ParWilsonFermionModule: ParWilsonFermion,fill!,Dx_serial!,project!,mul_dirac!,fermion_shift!,proj_halfspinor!,recon_halfspinor!,
            set_wing_fermi_correct!,proj_halfspinor_pt!,recon_halfspinor_pt!
using HDF5
using Base.Threads
n_threads = Threads.nthreads()
println("Using $(n_threads) thread(s).")

# Set up a gauge action
mq = -0.2450
κ = 1 / (2 * mq + 8)                            # hopping parameter
ferm_param = Actions.FermiActionParam_Wilson(κ, 1, 1e-16, 3000)
Nc = 3

# Different lattice sizes to test on
# Nx = 8; Ny = 8; Nz = 8; Nt = 8
Nx = 16; Ny = 16; Nz = 16; Nt = 48
# Nx = 32; Ny = 32; Nz = 32; Nt = 64
bc = ones(Int8, 4)

# initialize fermions
ser_ferm = WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc)
fill_mat = rand(eltype(ser_ferm.f), size(ser_ferm.f))
fill!(ser_ferm, fill_mat)
par_ferm = ParWilsonFermion(ser_ferm)

println("Dirac components of each fermion at (1, 1, 1, 1) in the first color index:")
println(ser_ferm.f[1, 2, 2, 2, 2, :])
println(par_ferm.f[1, 2, 2, 2, 2, :])

println("Dirac components of each fermion at (Nx, Ny, Nz, Nt) in the first color index:")
println(ser_ferm.f[1, Nx + 1, Ny + 1, Nz + 1, Nt + 1, :])
println(par_ferm.f[1, Nx + 1, Ny + 1, Nz + 1, Nt + 1, :])

################################################################################
############################## Test all projections ############################
################################################################################

# t_ser = Array{Float64}(undef, 2, 4)
# t_par = Array{Float64}(undef, 2, 4)
# println("Negative projections")
# for μ = 1 : 4
#     tmp1_ser = deepcopy(ser_ferm)
#     ser_out = WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc)
#     mul_dirac!(ser_out, ser_ferm.rminusγ[:, :, μ], tmp1_ser)
#     t_ser[1, μ] = @belapsed mul_dirac!($(tmp1_ser), ser_ferm.rminusγ[:, :, $(μ)], $(tmp1_ser))
#
#     par_out = deepcopy(par_ferm)
#     tmp1_par = deepcopy(par_ferm)
#     tmp_halfspinor = ParWilsonFermion(Nc, Nx, Ny, Nz, Nt, 2, ferm_param, bc)
#     project!(par_out, par_ferm, tmp_halfspinor, μ, false)
#     t_par[1, μ] = @belapsed project!($(tmp1_par), $(par_ferm), $(tmp_halfspinor), $(μ), false)
#
#     println("Difference for P^(-$(μ)) = $(maximum(abs.(ser_out.f - par_out.f)))")
# end
# println("Serial projection times: $(t_ser[1, :])")
# println("Parallel projection times: $(t_par[1, :])")

# println("Positive projections")
# for μ = 1 : 4
#     tmp1_ser = deepcopy(ser_ferm)
#     ser_out = WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc)
#     mul_dirac!(ser_out, ser_ferm.rplusγ[:, :, μ], tmp1_ser)
#     t_ser[2, μ] = @belapsed mul_dirac!($(tmp1_ser), ser_ferm.rplusγ[:, :, $(μ)], $(tmp1_ser))
#
#     par_out = deepcopy(par_ferm)
#     tmp1_par = deepcopy(par_ferm)
#     tmp_halfspinor = ParWilsonFermion(Nc, Nx, Ny, Nz, Nt, 2, ferm_param, bc)
#     project!(par_out, par_ferm, tmp_halfspinor, μ, true)
#     t_par[2, μ] = @belapsed project!($(tmp1_par), $(par_ferm), $(tmp_halfspinor), $(μ), true)
#
#     println("Difference for P^(+$(μ)) = $(maximum(abs.(ser_out.f - par_out.f)))")
# end
# println("Serial projection times: $(t_ser[2, :])")
# println("Parallel projection times: $(t_par[2, :])")

################################################################################
############################# Test single projection ###########################
################################################################################

μ = rand(1:4)
println("Projection for μ = $(μ)")
tmp1_ser = deepcopy(ser_ferm)
ser_out = WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc)
mul_dirac!(ser_out, ser_ferm.rminusγ[:, :, μ], tmp1_ser)
# Standard projection
t_proj_ser = @belapsed mul_dirac!(tmp1_ser, ser_ferm.rminusγ[:, :, μ], tmp1_ser)

par_out = deepcopy(par_ferm)
tmp1_par = deepcopy(par_ferm)
tmp_halfspinor = ParWilsonFermion(Nc, Nx, Ny, Nz, Nt, 2, ferm_param, bc)
project!(par_out, par_ferm, tmp_halfspinor, μ, false)
# Halfspinor projection
t_proj_par = @belapsed project!(tmp1_par, par_ferm, tmp_halfspinor, μ, false)

println("Difference for P^(-$(μ)) = $(maximum(abs.(ser_out.f - par_out.f)))")
println("Standard projection time: $(t_proj_ser)")
println("Halfspinor projection time: $(t_proj_par)")

################################################################################
############################ Test projection + shift ###########################
################################################################################

#=
NDW = 1
U = Array{SU3GaugeFields,1}(undef,4)
for μ = 1:4
    U[μ] = RandomGauges(Nc, Nx, Ny, Nz, Nt, NDW)
end

function shift_project!(out::WilsonFermion, x::WilsonFermion, tmp::WilsonFermion, U::Array{G,1}, μ::Integer) where G <: GaugeFields
    mul_dirac!(tmp, x.rminusγ[:, :, μ], x)              # project
    fermion_shift!(out, U, μ, tmp)                      # shift
end

function shift_project_par!(out::ParWilsonFermion, x::ParWilsonFermion, tmp::ParWilsonFermion, U::Array{G,1}, μ::Integer) where G <: GaugeFields
    proj_halfspinor!(tmp, x, μ, false)                  # project onto two-spinor
    fermion_shift!(tmp, U, μ, tmp)                      # shift halfspinor
    recon_halfspinor!(out, tmp, μ, false)               # recon halfspinor
end

shift_proj_ser = WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc)
shift_proj_par = ParWilsonFermion(shift_proj_ser)
tmp_ser = WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc)
tmp_par = ParWilsonFermion(Nc, Nx, Ny, Nz, Nt, 2, ferm_param, bc)

set_wing_fermi_correct!(ser_ferm)
set_wing_fermi_correct!(par_ferm)

t_shift_proj_ser = @belapsed shift_project!(shift_proj_ser, ser_ferm, tmp_ser, U, μ)
t_shift_proj_par = @belapsed shift_project_par!(shift_proj_par, par_ferm, tmp_par, U, μ)

set_wing_fermi_correct!(shift_proj_ser)
set_wing_fermi_correct!(shift_proj_par)

println("Standard shift + projection time: $(t_shift_proj_ser)")
println("Halfspinor shift + projection time: $(t_shift_proj_par)")
println("Difference for shift + project = $(maximum(abs.(shift_proj_ser.f - shift_proj_par.f)))")
=#

################################################################################
############################ Test projection at point ##########################
################################################################################

#=
μ = rand(1:4)
println("Testing projection at point vs broadcasting for μ = $(μ).")

function project_pts!(out::ParWilsonFermion, x::ParWilsonFermion, tmp_half::ParWilsonFermion, μ::Integer, plus::Bool)
    Nt = x.NT; Nz = x.NZ; Ny = x.NY; Nx = x.NX; Nc = x.NC
    for it = 1:Nt
        for iz = 1:Nz
            for iy = 1:Ny
                for ix = 1:Nx
                    # @simd for ic=1:a.NC
                    coords = @SVector [ix, iy, iz, it]
                    proj_halfspinor_pt!(tmp_half, x, coords, μ, plus)
                    recon_halfspinor_pt!(out, tmp_half, coords, μ, plus)
                end
            end
        end
    end
    return
end

function project_pts_threads!(out::ParWilsonFermion, x::ParWilsonFermion, tmp_half::ParWilsonFermion, μ::Integer, plus::Bool)
    Nt = x.NT; Nz = x.NZ; Ny = x.NY; Nx = x.NX; Nc = x.NC
    @threads for it = 1:Nt
        for iz = 1:Nz
            for iy = 1:Ny
                for ix = 1:Nx
                    # @simd for ic=1:a.NC
                    coords = @SVector [ix, iy, iz, it]
                    proj_halfspinor_pt!(tmp_half, x, coords, μ, plus)
                    recon_halfspinor_pt!(out, tmp_half, coords, μ, plus)
                end
            end
        end
    end
    return
end

par_all_out = deepcopy(par_ferm)
par_pt_out = deepcopy(par_ferm)
tmp_halfspinor = ParWilsonFermion(Nc, Nx, Ny, Nz, Nt, 2, ferm_param, bc)

t_proj_all = @belapsed project!(par_all_out, par_ferm, tmp_halfspinor, μ, false)
t_proj_pt = @belapsed project_pts!(par_pt_out, par_ferm, tmp_halfspinor, μ, false)
t_proj_pt_threads = @belapsed project_pts_threads!(par_pt_out, par_ferm, tmp_halfspinor, μ, false)

println("Difference for P^(-$(μ)) = $(maximum(abs.(par_all_out.f - par_pt_out.f)))")
println("Broadcast projection: $(t_proj_all)")
println("Point-by-point projection time: $(t_proj_pt)")
println("Point-by-point projection (with threads): $(t_proj_pt_threads)")
=#

################################################################################
################################## Save data ###################################
################################################################################

# if n_threads == 1
#     cmd = "w"                   # create new file
# else
#     cmd = "cw"                  # open old file
# end
# fout = h5open("/Users/theoares/parallel_lqcd/data/loop_order.h5", cmd)
# times = [t0, t1, t2, t3]
# # times = [t0, t1, t2, t3, t4]
# fout["nt$(n_threads)"] = times
# close(fout)
