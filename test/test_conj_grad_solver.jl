using BenchmarkTools
using LinearAlgebra
using LatticeQCD.Actions:Actions
using LatticeQCD.WilsonFermion_module:WilsonFermion,LinearAlgebra.rmul!,Wx!

include("../src/fermions/ParWilsonFermion.jl")
include("../src/solvers/ConjGradientSolver.jl")
using ..ParWilsonFermionModule: ParWilsonFermion,LinearAlgebra.mul!
print("Loaded ParWilsonFermionModule") 
using ..ConjGradientSolverModule: solve!
#print("Loaded ConjGradientSolver_module") 

# Set up a gauge action
mq = -0.2450
κ = 1 / (2 * mq + 8)                            # hopping parameter
ferm_param = Actions.FermiActionParam_Wilson(κ, 1, 1e-16, 3000)
Actions.show_parameters_action(ferm_param)
Nc = 3
Nx = 1; Ny = 1; Nz = 1; Nt = 2
bc = ones(Int8, 1)
mul_factor = 4.3

Nlattice=Nc*Nx*Ny*Nz*Nt
Ntotal=4*Nlattice

rank=3
rank=min(rank,Nlattice)
println("Testing on 4x", Nlattice,"x",Nlattice," matrix of rank ", rank)


semi_positive_def_matrix=A = zeros(Nlattice,Nlattice)

for i=1:rank
    v=randn(Nlattice)
    lambda=100*rand()
    global semi_positive_def_matrix=semi_positive_def_matrix+lambda*v*v'
end

the_matrix=[[semi_positive_def_matrix,semi_positive_def_matrix,semi_positive_def_matrix,semi_positive_def_matrix]  [semi_positive_def_matrix,semi_positive_def_matrix,semi_positive_def_matrix,semi_positive_def_matrix] [semi_positive_def_matrix,semi_positive_def_matrix,semi_positive_def_matrix,semi_positive_def_matrix] [semi_positive_def_matrix,semi_positive_def_matrix,semi_positive_def_matrix,semi_positive_def_matrix] ]

ser_ferm = WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc)
par_ferm = ParWilsonFermion(ser_ferm)                           # can construct either way
# Test __rmul__
out_ferm=ParWilsonFermion(ser_ferm)
fill!(par_ferm,1)
fill!(out_ferm,0)

workspace_ferm=out_ferm+par_ferm

solve!(semi_positive_def_matrix, par_ferm, out_ferm)
mul!(workspace_ferm,semi_positive_def_matrix,par_ferm)
test_zero=out_ferm - workspace_ferm

print("Norm of residual: ", sqrt(test_zero*test_zero))


