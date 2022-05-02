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


function LinearAlgebra.mul1!(xout::ParWilsonFermion,A::AbstractMatrix,x::ParWilsonFermion)
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    NC = x.NC
    for ic=1:NC
        for it=1:NT
            for iz=1:NZ
                for iy=1:NY
                    @simd for ix=1:NX
                            e1 = x[ic,ix,iy,iz,it,1]
                            e2 = x[ic,ix,iy,iz,it,2]
                            e3 = x[ic,ix,iy,iz,it,3]
                            e4 = x[ic,ix,iy,iz,it,4]

                            xout[ic,ix,iy,iz,it,1] = A[1,1]*e1+A[1,2]*e2+A[1,3]*e3+A[1,4]*e4
                            xout[ic,ix,iy,iz,it,2] = A[2,1]*e1+A[2,2]*e2+A[2,3]*e3+A[2,4]*e4
                            xout[ic,ix,iy,iz,it,3] = A[3,1]*e1+A[3,2]*e2+A[3,3]*e3+A[3,4]*e4
                            xout[ic,ix,iy,iz,it,4] = A[4,1]*e1+A[4,2]*e2+A[4,3]*e3+A[4,4]*e4

                    end
                end
            end
        end
    end
end

function LinearAlgebra.mul1!(xout::ParWilsonFermion,A::AbstractMatrix,x::ParWilsonFermion)
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    NC = x.NC
    for ic=1:NC
        for it=1:NT
            for iz=1:NZ
                for iy=1:NY
                    @simd for ix=1:NX

                        for d1 =  1:4
                            for d2 = 1:4
                                xout[ic,ix,iy,iz,it,d1] += A[d1,d2]*x[ic,ix,iy,iz,it,d2]
                            end
                        end

                    end
                end
            end
        end
    end
end

function LinearAlgebra.mul1!(xout::ParWilsonFermion,x::ParWilsonFermion,A::AbstractMatrix)
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    NC = x.NC

    for ic=1:NC
        for it=1:NT
            for iz=1:NZ
                for iy=1:NY
                    @simd for ix=1:NX
                        e1 = x[ic,ix,iy,iz,it,1]
                        e2 = x[ic,ix,iy,iz,it,2]
                        e3 = x[ic,ix,iy,iz,it,3]
                        e4 = x[ic,ix,iy,iz,it,4]

                        xout[ic,ix,iy,iz,it,1] = A[1,1]*e1+A[2,1]*e2+A[3,1]*e3+A[4,1]*e4
                        xout[ic,ix,iy,iz,it,2] = A[1,2]*e1+A[2,2]*e2+A[3,2]*e3+A[4,2]*e4
                        xout[ic,ix,iy,iz,it,3] = A[1,3]*e1+A[2,3]*e2+A[3,3]*e3+A[4,3]*e4
                        xout[ic,ix,iy,iz,it,4] = A[1,4]*e1+A[2,4]*e2+A[3,4]*e3+A[4,4]*e4
                    end
                end
            end
        end
    end
end

function LinearAlgebra.mul2!(xout::ParWilsonFermion,x::ParWilsonFermion,A::AbstractMatrix)
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    NC = x.NC

    for ic=1:NC
        for it=1:NT
            for iz=1:NZ
                for iy=1:NY
                    @simd for ix=1:NX

                        for d1 =  1:4
                            for d2 = 1:4
                                xout[ic,ix,iy,iz,it,d1] += A[d1,d2]*x[ic,ix,iy,iz,it,d2]
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
t1 = @belapsed mul1!(par_ferm, mul_factor)
t2 = @belapsed mul2!(par_ferm, mul_factor)

println("Time for parallel loop 1: $(t1) seconds")
println("Time for parallel loop 2: $(t2) seconds")

