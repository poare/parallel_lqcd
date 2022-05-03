#=
This script will time the difference between mutating a Julia array in either its first or last
index, respectively. For example,

julia> @btime ser_ferm.f[2, :, :, :, :, :] .= 3;
  3.638 ms (7 allocations: 400 bytes)

julia> @btime ser_ferm.f[:, :, :, :, :, 2] .= 3;
  504.809 μs (7 allocations: 400 bytes)
=#

using BenchmarkTools
NC = 3; ND = 4; NX = 16; NY = 16; NZ = 16; NT = 48
f1 = rand(ComplexF64, NC, NX+2, NY+2, NZ+2, NT+2, ND)           # default format
f2 = rand(ComplexF64, ND, NX+2, NY+2, NZ+2, NT+2, NC)           # modified format

println("Default fermion format:")
@btime f1[:, :, :, :, :, 1] .= 3;

println("Modified fermion format:")
@btime f2[1, :, :, :, :, :] .= 3;
