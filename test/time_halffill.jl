#=
This script will time the difference between filling a Julia array by broadcasting,
and filling it by explicitly looping over the indices.
=#

# using Base.Threads

using BenchmarkTools
NC = 3; ND = 4; NX = 16; NY = 16; NZ = 16; NT = 48
f1 = rand(ComplexF64, NC, NX+2, NY+2, NZ+2, NT+2, ND)
f2 = rand(ComplexF64, NC, NX+2, NY+2, NZ+2, NT+2, ND)
function fill_half!(x::Array{ComplexF64,6}, y::Array{ComplexF64,6})
    x[:, :, :, :, :, 1] .= y[:, :, :, :, :, 1]
    x[:, :, :, :, :, 2] .= y[:, :, :, :, :, 2]
    return
end

function fill_half_views!(x::Array{ComplexF64,6}, y::Array{ComplexF64,6})
    @views x[:, :, :, :, :, 1] .= y[:, :, :, :, :, 1]
    @views x[:, :, :, :, :, 2] .= y[:, :, :, :, :, 2]
    return
end

function fill_half_loop!(x::Array{ComplexF64,6}, y::Array{ComplexF64,6})
    @inbounds for α=1:2
        @inbounds for it=1:NT
            @inbounds for iz=1:NZ
                @inbounds for iy=1:NY
                    @inbounds for ix=1:NX
                        @inbounds for ic=1:NC
                            x[ic,ix,iy,iz,it,α] = y[ic,ix,iy,iz,it,α]
                        end
                    end
                end
            end
        end
    end
    return
end

println("Direct broadcasting:")
@btime fill_half!(f1, f2)

println("Direct broadcasting with views:")
f1 = rand(ComplexF64, NC, NX+2, NY+2, NZ+2, NT+2, ND)
@btime fill_half_views!(f1, f2)

println("Loop broadcasting")
f1 = rand(ComplexF64, NC, NX+2, NY+2, NZ+2, NT+2, ND)
@btime fill_half_loop!(f1, f2)

# function fill_half_back!(x::Array{ComplexF64,6}, y::Array{ComplexF64,6})
#     for ic=1:NC
#         for ix=1:NX
#             for iy=1:NY
#                 for iz=1:NZ
#                     for it=1:NT
#                         for α=1:2
#                             x[ic,ix,iy,iz,it,α] = y[ic,ix,iy,iz,it,α]
#                         end
#                     end
#                 end
#             end
#         end
#     end
#     return
# end
#
# function fill_half_loop_threads!(x::Array{ComplexF64,6}, y::Array{ComplexF64,6})
#     @threads for α=1:2
#         for it=1:NT
#             for iz=1:NZ
#                 for iy=1:NY
#                     for ix=1:NX
#                         @simd for ic=1:NC
#                             x[ic,ix,iy,iz,it,α] = y[ic,ix,iy,iz,it,α]
#                         end
#                     end
#                 end
#             end
#         end
#     end
#     return
# end
#
# println("Loop broadcasting (other direction)")
# f1 = rand(ComplexF64, NC, NX+2, NY+2, NZ+2, NT+2, ND)
# @btime fill_half_back!(f1, f2)
#
# println("Loop broadcasting with threading")
# f1 = rand(ComplexF64, NC, NX+2, NY+2, NZ+2, NT+2, ND)
# @btime fill_half_loop_threads!(f1, f2)
