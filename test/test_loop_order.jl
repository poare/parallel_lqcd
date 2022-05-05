#=
This script is testing a few different orders of parallelization for the loops in ParWilsonFermion and
trying to find the optimal one. Each loop is of the same order and has to loop over (α, x, y, z, t, a).
Here α is a Dirac index and is either in {1, 2} or {1, 2, 3, 4}; x, y, z, t are lattice site positions, and
values in {0, 1, ..., L - 1} (or T - 1 for t); a is a color index and valued in {1, 2, 3}.
=#

using BenchmarkTools
using LatticeQCD.Actions:Actions
using LatticeQCD.WilsonFermion_module:WilsonFermion,LinearAlgebra.rmul!,Wx!
include("../src/fermions/ParWilsonFermion.jl")
using ..ParWilsonFermionModule: ParWilsonFermion,fill!
using Base.Threads
using HDF5
n_threads = Threads.nthreads()
println("Using $(n_threads) thread(s).")

# Set up a gauge action
mq = -0.2450
κ = 1 / (2 * mq + 8)                            # hopping parameter
ferm_param = Actions.FermiActionParam_Wilson(κ, 1, 1e-16, 3000)
# Actions.show_parameters_action(ferm_param)
Nc = 3
# Nx = 8; Ny = 8; Nz = 8; Nt = 8
Nx = 16; Ny = 16; Nz = 16; Nt = 48
bc = ones(Int8, 4)
mul_factor = 4.3

# initialize fermions
ser_ferm = WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc)
fill_mat = rand(eltype(ser_ferm.f), size(ser_ferm.f))
fill!(ser_ferm, fill_mat)
par_ferm = ParWilsonFermion(ser_ferm)                           # can construct either way

################################################################################
############################### Init and run test ##############################
################################################################################

# setup rmul functions with loop parallelization done in a different order
function rmul1!(a::ParWilsonFermion,b::T) where T <: Number
    @threads for α=1:a.ND
        for it=1:a.NT
            for iz=1:a.NZ
                for iy=1:a.NY
                    for ix=1:a.NX
                        @simd for ic=1:a.NC
                            a[ic,ix,iy,iz,it,α] = b*a[ic,ix,iy,iz,it,α]
                        end
                    end
                end
            end
        end
    end
    return a
end

function rmul2!(a::ParWilsonFermion,b::T) where T <: Number
    @threads for it=1:a.NT
        for α=1:a.ND
            for iz=1:a.NZ
                for iy=1:a.NY
                    for ix=1:a.NX
                        @simd for ic=1:a.NC
                            a[ic,ix,iy,iz,it,α] = b*a[ic,ix,iy,iz,it,α]
                        end
                    end
                end
            end
        end
    end
    return a
end

function rmul3!(a::ParWilsonFermion,b::T) where T <: Number
    @threads for it=1:a.NT
        for α=1:a.ND
            for iz=1:a.NZ
                for iy=1:a.NY
                    for ic=1:a.NC
                        @simd for ix=1:a.NX
                            a[ic,ix,iy,iz,it,α] = b*a[ic,ix,iy,iz,it,α]
                        end
                    end
                end
            end
        end
    end
    return a
end

function rmul4!(a::ParWilsonFermion,b::T) where T <: Number
    t = @spawn for it=1:a.NT
        for α=1:a.ND
            for iz=1:a.NZ
                for iy=1:a.NY
                    for ix=1:a.NX
                        @simd for ic=1:a.NC
                            a[ic,ix,iy,iz,it,α] = b*a[ic,ix,iy,iz,it,α]
                        end
                    end
                end
            end
        end
    end
    wait(t)
    return a
end

function rmul_broadcast!(a::ParWilsonFermion,b::T) where T <: Number
    a.f[:, :, :, :, :, :] .= b .* a.f[:, :, :, :, :, :]
    # @threads for α=1:a.ND
    #     for it=1:a.NT
    #         for iz=1:a.NZ
    #             for iy=1:a.NY
    #                 for ix=1:a.NX
    #                     @simd for ic=1:a.NC
    #                         a[ic,ix,iy,iz,it,α] = b*a[ic,ix,iy,iz,it,α]
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end
    return a
end

# function rmul5!(a::ParWilsonFermion,b::T) where T <: Number
#     @threads for it=1:a.NT
#         for α=1:a.ND
#             for iz=1:a.NZ
#                 for iy=1:a.NY
#                     for ic=1:a.NC
#                         @simd for ix=1:a.NX
#                             @inbounds a[ic,ix,iy,iz,it,α] = b*a[ic,ix,iy,iz,it,α]
#                         end
#                     end
#                 end
#             end
#         end
#     end
#     return a
# end

# Test __rmul__
println("Testing __rmul__ overloading.")
# may want to copy to make sure they're using the same data
t0 = @belapsed rmul!(ser_ferm, mul_factor)
t1 = @belapsed rmul1!(par_ferm, mul_factor)
t2 = @belapsed rmul2!(par_ferm, mul_factor)
t3 = @belapsed rmul3!(par_ferm, mul_factor)
t4 = @belapsed rmul_broadcast!(par_ferm, mul_factor)
# t5 = @belapsed rmul5!(par_ferm, mul_factor)

println("Time for serial loop: $(t0) seconds")
println("Time for parallel loop 1: $(t1) seconds")
println("Time for parallel loop 2: $(t2) seconds")
println("Time for parallel loop 3: $(t3) seconds")
println("Time for broadcasted loop: $(t4) seconds")
# println("Time for parallel loop 4: $(t4) seconds")
# println("Time for parallel loop 5: $(t5) seconds")

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
