using BenchmarkTools
using LinearAlgebra
using LatticeQCD.Actions:Actions
using LatticeQCD.WilsonFermion_module:WilsonFermion,LinearAlgebra.rmul!,Wx!

include("../src/fermions/ParWilsonFermion.jl")
include("../src/solvers/ConjGradientSolver.jl")
using ..ParWilsonFermionModule: ParWilsonFermion,LinearAlgebra.mul!
print("Loaded ParWilsonFermionModule")
using ..ConjGradientSolverModule: solve!,solve_verbose!
#print("Loaded ConjGradientSolver_module")

# Set up a gauge action
mq = -0.2450
κ = 1 / (2 * mq + 8)                            # hopping parameter
ferm_param = Actions.FermiActionParam_Wilson(κ, 1, 1e-16, 3000)
Actions.show_parameters_action(ferm_param)
Nc = 3
Nx = 2; Ny = 2; Nz = 2; Nt = 2
bc = ones(Int8, 4)
mul_factor = 4.3

Nlattice=Nc*Nx*Ny*Nz*Nt
Ntotal=4*Nlattice

rank=15
rank=min(rank,Nlattice)
println("Testing on (4x", Nlattice,")^2 matrix of rank ", rank)


semi_positive_def_matrix_flattened= zeros(Nlattice,Nlattice)

for i=1:rank
    v=randn(Nlattice)
    lambda=100*rand()
    global semi_positive_def_matrix_flattened+=lambda*v*v'
end
semi_positive_def_matrix=


the_matrix=[[semi_positive_def_matrix,semi_positive_def_matrix,semi_positive_def_matrix,semi_positive_def_matrix]  [semi_positive_def_matrix,semi_positive_def_matrix,semi_positive_def_matrix,semi_positive_def_matrix] [semi_positive_def_matrix,semi_positive_def_matrix,semi_positive_def_matrix,semi_positive_def_matrix] [semi_positive_def_matrix,semi_positive_def_matrix,semi_positive_def_matrix,semi_positive_def_matrix] ]

println(size(the_matrix[1,1]))

ser_ferm = WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc)
par_ferm = ParWilsonFermion(ser_ferm)                           # can construct either way
# Test __rmul__
out_ferm=ParWilsonFermion(ser_ferm)
fill!(par_ferm,1)
fill!(out_ferm,0)

workspace_ferm=out_ferm+par_ferm

solve_verbose!(the_matrix, par_ferm, out_ferm)
mul!(workspace_ferm,the_matrix,out_ferm)
println(out_ferm)
test_zero=par_ferm - workspace_ferm

test_error=sqrt(test_zero*test_zero)
println("Norm of residual difference of Dx-b: ", sqrt(test_zero*test_zero))
println("Mean error per element of Dx-b: ", test_error^2/Ntotal)
