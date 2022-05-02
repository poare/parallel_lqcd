using BenchmarkTools
using LatticeQCD.Actions:Actions
using LatticeQCD.WilsonFermion_module:WilsonFermion,LinearAlgebra.rmul!,Wx!
include("../src/fermions/ParWilsonFermion.jl")
using ..ParWilsonFermionModule: ParWilsonFermion
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
par_ferm = ParWilsonFermion(ser_ferm)                           # can construct either way

################################################################################
############################### Init and run test ##############################
################################################################################

############################################################################
########################### Shifting operations ############################
############################################################################

#=
These are gauge invariant shifting operations. They'll likely be useful to reimplement: notice that the type
T here is a gauge field, so u::Array{T, 1} is just a gauge field u[μ], where μ ∈ {1, 2, 3, 4}: the multiplication
by u[μ] serves to make the shifted gauge field invariant. We'll likely need to implement all these functions for
ParWilsonFermions.
=#


#=
Comparing fermion_shift! and fermion_shift_gamma!
=#

function fermion_shift1!(b::ParWilsonFermion,u::Array{T,1},μ::Int,a::ParWilsonFermion) where T <: SU3GaugeFields
    if μ == 0
        substitute!(b,a)
        return
    end

    NX = a.NX
    NY = a.NY
    NZ = a.NZ
    NT = a.NT
    ND = a.ND
    NC = 3

    if μ > 0
        n6 = size(a.f)[6]
        for ialpha=1:ND
            for it=1:NT
                it1 = it + ifelse(μ ==4,1,0)
                for iz=1:NZ
                    iz1 = iz + ifelse(μ == 3,1,0)
                    for iy=1:NY
                        iy1 = iy + ifelse(μ == 2,1,0)
                        @simd for ix=1:NX
                            ix1 = ix + ifelse(μ ==1,1,0)

                            b[1,ix,iy,iz,it,ialpha] =   u[μ][1,1,ix,iy,iz,it]*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                        u[μ][1,2,ix,iy,iz,it]*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                        u[μ][1,3,ix,iy,iz,it]*a[3,ix1,iy1,iz1,it1,ialpha]

                            b[2,ix,iy,iz,it,ialpha] =   u[μ][2,1,ix,iy,iz,it]*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                        u[μ][2,2,ix,iy,iz,it]*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                        u[μ][2,3,ix,iy,iz,it]*a[3,ix1,iy1,iz1,it1,ialpha]

                            b[3,ix,iy,iz,it,ialpha] =   u[μ][3,1,ix,iy,iz,it]*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                        u[μ][3,2,ix,iy,iz,it]*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                        u[μ][3,3,ix,iy,iz,it]*a[3,ix1,iy1,iz1,it1,ialpha]
                        end
                    end
                end
            end
        end

    elseif μ < 0
        for ialpha =1:ND
            for it=1:NT
                it1 = it - ifelse(-μ ==4,1,0)
                for iz=1:NZ
                    iz1 = iz - ifelse(-μ ==3,1,0)
                    for iy=1:NY
                        iy1 = iy - ifelse(-μ ==2,1,0)
                        @simd for ix=1:NX
                            ix1 = ix - ifelse(-μ ==1,1,0)

                            b[1,ix,iy,iz,it,ialpha] = conj(u[-μ][1,1,ix1,iy1,iz1,it1])*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                        conj(u[-μ][2,1,ix1,iy1,iz1,it1])*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                        conj(u[-μ][3,1,ix1,iy1,iz1,it1])*a[3,ix1,iy1,iz1,it1,ialpha]

                            b[2,ix,iy,iz,it,ialpha] = conj(u[-μ][1,2,ix1,iy1,iz1,it1])*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                        conj(u[-μ][2,2,ix1,iy1,iz1,it1])*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                        conj(u[-μ][3,2,ix1,iy1,iz1,it1])*a[3,ix1,iy1,iz1,it1,ialpha]

                            b[3,ix,iy,iz,it,ialpha] = conj(u[-μ][1,3,ix1,iy1,iz1,it1])*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                        conj(u[-μ][2,3,ix1,iy1,iz1,it1])*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                        conj(u[-μ][3,3,ix1,iy1,iz1,it1])*a[3,ix1,iy1,iz1,it1,ialpha]

                        end
                    end
                end
            end
        end
    end

end

function fermion_shift2!(b::ParWilsonFermion,u::Array{T,1},μ::Int,a::ParWilsonFermion) where T <: SU3GaugeFields
    if μ == 0
        substitute!(b,a)
        return
    end

    NX = a.NX
    NY = a.NY
    NZ = a.NZ
    NT = a.NT
    ND = a.ND
    NC = 3

    if μ > 0
        n6 = size(a.f)[6]
        for ialpha=1:ND
            for it=1:NT
                it1 = it + ifelse(μ ==4,1,0)
                for iz=1:NZ
                    iz1 = iz + ifelse(μ == 3,1,0)
                    for iy=1:NY
                        iy1 = iy + ifelse(μ == 2,1,0)
                        @simd for ix=1:NX
                            ix1 = ix + ifelse(μ ==1,1,0)

                            for n1 = 1:3
                                for n2 = 1:3
                                    n1,ix,iy,iz,it,ialpha] += u[μ][n1,n2,ix,iy,iz,it]*a[n2,ix1,iy1,iz1,it1,ialpha]
                                end
                            end

                        end
                    end
                end
            end
        end

    elseif μ < 0
        for ialpha =1:ND
            for it=1:NT
                it1 = it - ifelse(-μ ==4,1,0)
                for iz=1:NZ
                    iz1 = iz - ifelse(-μ ==3,1,0)
                    for iy=1:NY
                        iy1 = iy - ifelse(-μ ==2,1,0)
                        @simd for ix=1:NX
                            ix1 = ix - ifelse(-μ ==1,1,0)

                            for n1 = 1:3
                                for n2 = 1:3 
                                    b[n1,ix,iy,iz,it,ialpha] += conj(u[μ][n1,n2,ix,iy,iz,it])*a[n2,ix1,iy1,iz1,it1,ialpha]
                                end
                            end

                        end
                    end
                end
            end
        end
    end

end


function fermion_shift_gamma1!(b::ParWilsonFermion,u::Array{T,1},μ::Int,a::ParWilsonFermion) where T <: SU3GaugeFields
    if μ == 0
        substitute!(b,a)
        return
    end

    NX = a.NX
    NY = a.NY
    NZ = a.NZ
    NT = a.NT
    NC = 3

    if μ > 0
        n6 = size(a.f)[6]
        for ialpha=1:ND
            for it=1:NT
                it1 = it + ifelse(μ ==4,1,0)
                for iz=1:NZ
                    iz1 = iz + ifelse(μ ==3,1,0)
                    for iy=1:NY
                        iy1 = iy + ifelse(μ ==2,1,0)
                        @simd for ix=1:NX
                            ix1 = ix + ifelse(μ ==1,1,0)

                            b[1,ix,iy,iz,it,ialpha] = u[μ][1,1,ix,iy,iz,it]*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                        u[μ][1,2,ix,iy,iz,it]*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                        u[μ][1,3,ix,iy,iz,it]*a[3,ix1,iy1,iz1,it1,ialpha]

                            b[2,ix,iy,iz,it,ialpha] = u[μ][2,1,ix,iy,iz,it]*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                        u[μ][2,2,ix,iy,iz,it]*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                        u[μ][2,3,ix,iy,iz,it]*a[3,ix1,iy1,iz1,it1,ialpha]

                            b[3,ix,iy,iz,it,ialpha] = u[μ][3,1,ix,iy,iz,it]*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                        u[μ][3,2,ix,iy,iz,it]*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                        u[μ][3,3,ix,iy,iz,it]*a[3,ix1,iy1,iz1,it1,ialpha]

                        end
                    end
                end
            end
        end

    elseif μ < 0
        for ialpha =1:ND
            for it=1:NT
                it1 = it - ifelse(-μ ==4,1,0) #idel[4]
                for iz=1:NZ
                    iz1 = iz - ifelse(-μ ==3,1,0) #idel[3]
                    for iy=1:NY
                        iy1 = iy - ifelse(-μ ==2,1,0)  #idel[2]
                        @simd for ix=1:NX
                            ix1 = ix - ifelse(-μ ==1,1,0) #idel[1]

                            b[1,ix,iy,iz,it,ialpha] = conj(u[-μ][1,1,ix1,iy1,iz1,it1])*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                        conj(u[-μ][2,1,ix1,iy1,iz1,it1])*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                        conj(u[-μ][3,1,ix1,iy1,iz1,it1])*a[3,ix1,iy1,iz1,it1,ialpha]

                            b[2,ix,iy,iz,it,ialpha] = conj(u[-μ][1,2,ix1,iy1,iz1,it1])*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                        conj(u[-μ][2,2,ix1,iy1,iz1,it1])*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                        conj(u[-μ][3,2,ix1,iy1,iz1,it1])*a[3,ix1,iy1,iz1,it1,ialpha]

                            b[3,ix,iy,iz,it,ialpha] = conj(u[-μ][1,3,ix1,iy1,iz1,it1])*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                        conj(u[-μ][2,3,ix1,iy1,iz1,it1])*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                        conj(u[-μ][3,3,ix1,iy1,iz1,it1])*a[3,ix1,iy1,iz1,it1,ialpha]

                        end
                    end
                end
            end
        end
    end
end

function fermion_shift_gamma!(b::ParWilsonFermion,u::Array{T,1},μ::Int,a::ParWilsonFermion) where T <: SU3GaugeFields
    if μ == 0
        substitute!(b,a)
        return
    end

    NX = a.NX
    NY = a.NY
    NZ = a.NZ
    NT = a.NT
    NC = 3

    if μ > 0
        n6 = size(a.f)[6]
        for ialpha=1:ND
            for it=1:NT
                it1 = it + ifelse(μ ==4,1,0)
                for iz=1:NZ
                    iz1 = iz + ifelse(μ ==3,1,0)
                    for iy=1:NY
                        iy1 = iy + ifelse(μ ==2,1,0)
                        @simd for ix=1:NX
                            ix1 = ix + ifelse(μ ==1,1,0)
                            for n1 = 1:3
                                for n2 in 1:3 
                                    b[n1,ix,iy,iz,it,ialpha] += u[μ][n1,n2,ix,iy,iz,it]*a[n2,ix1,iy1,iz1,it1,ialpha]
                                end
                            end

                        end
                    end
                end
            end
        end

    elseif μ < 0
        for ialpha =1:ND
            for it=1:NT
                it1 = it - ifelse(-μ ==4,1,0) #idel[4]
                for iz=1:NZ
                    iz1 = iz - ifelse(-μ ==3,1,0) #idel[3]
                    for iy=1:NY
                        iy1 = iy - ifelse(-μ ==2,1,0)  #idel[2]
                        @simd for ix=1:NX
                            ix1 = ix - ifelse(-μ ==1,1,0) #idel[1]
                            for n1 = 1:3
                                for n2 = 1:3 
                                    b[n1,ix,iy,iz,it,ialpha] += u[μ][n1,n2,ix,iy,iz,it]*a[n2,ix1,iy1,iz1,it1,ialpha]
                                end
                            end

                        end
                    end
                end
            end
        end
    end
end



# Test __mul__
println("Testing __mul__ overloading.")
# may want to copy to make sure they're using the same data
t1 = @belapsed fermion_shift1!(par_ferm, mul_factor)
t2 = @belapsed fermion_shift2!(par_ferm, mul_factor)

t3 = @belapsed fermion_shift_gamma1!(par_ferm, mul_factor)
t4 = @belapsed fermion_shift_gamma2!(par_ferm, mul_factor)

println("Time for parallel loop 1: $(t1) seconds")
println("Time for parallel loop 2: $(t2) seconds")

println("Time for parallel loop 1: $(t3) seconds")
println("Time for parallel loop 2: $(t4) seconds")
