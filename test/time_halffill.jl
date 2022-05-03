#=
This script will time the difference between filling a Julia array by broadcasting,
and filling it by explicitly looping over the indices.
=#

using BenchmarkTools
using Base.Threads
NC = 3; ND = 4; NX = 16; NY = 16; NZ = 16; NT = 48
f1 = rand(ComplexF64, NC, NX+2, NY+2, NZ+2, NT+2, ND)
f2 = rand(ComplexF64, NC, NX+2, NY+2, NZ+2, NT+2, ND)

function fill_half!(x::Array{ComplexF64,6}, y::Array{ComplexF64,6})
    x[:, :, :, :, :, 1] .= y[:, :, :, :, :, 1]
    x[:, :, :, :, :, 2] .= y[:, :, :, :, :, 2]
    x[:, :, :, :, :, 3] .= im .* y[:, :, :, :, :, 3]
    x[:, :, :, :, :, 4] .= -im * y[:, :, :, :, :, 4]
    return
end

function fill_half_loop!(x::Array{ComplexF64,6}, y::Array{ComplexF64,6})
    for α=1:2
        for it=1:NT
            for iz=1:NZ
                for iy=1:NY
                    for ix=1:NX
                        @simd for ic=1:NC
                            x[ic,ix,iy,iz,it,α] = x[ic,ix,iy,iz,it,α]
                        end
                    end
                end
            end
        end
    end
    x[:, :, :, :, :, 3] .= im .* y[:, :, :, :, :, 3]
    x[:, :, :, :, :, 4] .= -im * y[:, :, :, :, :, 4]
    return
end

function fill_half_loop_threads!(x::Array{ComplexF64,6}, y::Array{ComplexF64,6})
    @threads for α=1:2
        for it=1:NT
            for iz=1:NZ
                for iy=1:NY
                    for ix=1:NX
                        @simd for ic=1:NC
                            x[ic,ix,iy,iz,it,α] = x[ic,ix,iy,iz,it,α]
                        end
                    end
                end
            end
        end
    end
    x[:, :, :, :, :, 3] .= im .* y[:, :, :, :, :, 3]
    x[:, :, :, :, :, 4] .= -im * y[:, :, :, :, :, 4]
    return
end

println("Direct broadcasting:")
@btime fill_half!(f1, f2)

println("Loop broadcasting")
f1 = rand(ComplexF64, NC, NX+2, NY+2, NZ+2, NT+2, ND)
@btime fill_half_loop!(f1, f2)

println("Loop broadcasting with threading")
f1 = rand(ComplexF64, NC, NX+2, NY+2, NZ+2, NT+2, ND)
@btime fill_half_loop_threads!(f1, f2)
