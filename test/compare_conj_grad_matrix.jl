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

Ntotal_glob=300

rank_glob=2*Ntotal_glob
#rank=min(rank,Ntotal)
println("Testing on ", Ntotal_glob,"x",Ntotal_glob," matrix of rank ", rank)


semi_positive_def_matrix_flattened_glob = zeros(Ntotal_glob,Ntotal_glob)

for i=1:rank_glob
    v=randn(Ntotal_glob)
    lambda=100*rand()
    global semi_positive_def_matrix_flattened_glob += lambda * v*v'
end


the_matrix_glob=semi_positive_def_matrix_flattened_glob

par_ferm_glob = Vector{Float64}(undef, Ntotal_glob)                      
out_ferm_glob=Vector{Float64}(undef, Ntotal_glob)    
workspace_ferm_glob=Vector{Float64}(undef, Ntotal_glob)
workspace_ferm2_glob=Vector{Float64}(undef, Ntotal_glob)

fill!(par_ferm_glob,1)
fill!(out_ferm_glob,0)
fill!(workspace_ferm_glob,0)
fill!(workspace_ferm2_glob,0)

@time solve!(the_matrix_glob, par_ferm_glob, out_ferm_glob)
@time workspace_ferm2_glob=the_matrix_glob \ par_ferm_glob

for Ntotal in [5,10,20,50,100,200,500,1000]
    println("Ntotal=",Ntotal)
    for j=1:100

        rank=2*Ntotal
        #println("Testing on ", Ntotal,"x",Ntotal," matrix of rank ", rank)


        the_matrix = zeros(Ntotal,Ntotal)

        for i=1:rank
            v=randn(Ntotal)
            lambda=100*rand()
            the_matrix += lambda * v*v'
        end

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

    end

end
