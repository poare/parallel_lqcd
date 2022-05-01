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

Ntotal=300

rank=2*Ntotal
#rank=min(rank,Ntotal)
println("Testing on ", Ntotal,"x",Ntotal," matrix of rank ", rank)


semi_positive_def_matrix_flattened = zeros(Ntotal,Ntotal)

for i=1:rank
    v=randn(Ntotal)
    lambda=100*rand()
    global semi_positive_def_matrix_flattened += lambda * v*v'
end


the_matrix=semi_positive_def_matrix_flattened

par_ferm = Vector{Float64}(undef, Ntotal)                      
out_ferm=Vector{Float64}(undef, Ntotal)    
workspace_ferm=Vector{Float64}(undef, Ntotal)
workspace_ferm2=Vector{Float64}(undef, Ntotal)

fill!(par_ferm,1)
fill!(out_ferm,0)
fill!(workspace_ferm,0)
fill!(workspace_ferm2,0)

@time solve!(the_matrix, par_ferm, out_ferm)
@time workspace_ferm2=the_matrix \ par_ferm

fill!(par_ferm,1)
fill!(out_ferm,0)
fill!(workspace_ferm,0)
fill!(workspace_ferm2,0)

@time solve!(the_matrix, par_ferm, out_ferm)
@time workspace_ferm2=the_matrix \ par_ferm

mul!(workspace_ferm,the_matrix,out_ferm)

test_zero=par_ferm - workspace_ferm

test_error=sqrt(test_zero*test_zero)
println("Norm of residual difference of Dx-b: ", sqrt(test_zero*test_zero))
println("Mean error per element of Dx-b: ", test_error^2/Ntotal)



