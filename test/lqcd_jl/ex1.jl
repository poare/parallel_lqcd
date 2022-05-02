using LatticeQCD.Actions:Actions
using LatticeQCD.WilsonFermion_module:WilsonFermion,Wx!
# using LatticeQCD.Actions:FermiActionParam,FermiActionParam_Wilson,show_parameters_action

# Set up a gauge action
mq = -0.2450
κ = 1 / (2 * mq + 8)                            # hopping parameter
ferm_param = Actions.FermiActionParam_Wilson(κ, 1, 1e-16, 3000)
Actions.show_parameters_action(ferm_param)

Nc = 3
Nx = 8; Ny = 8; Nz = 8; Nt = 8
bc = ones(Int8, 4)
ferm = WilsonFermion(Nc, Nx, Ny, Nz, Nt, ferm_param, bc)
println(size(ferm))                             # Julia's size function is overloaded to give the dimensions
ferm[1, 0, 0, 0, 0, 1] = rand()                 # Spacetime coordinates are 0-indexed, color / spinor indices are 1 indexed
print(ferm[1, 0, 0, 0, 0, 1])
