# parallel_lqcd
Julia code for parallelizing lattice QCD code. Some papers to check out that I think will be helpful (all in the Dropbox):
- [High-Performance Parallelism Pearls Chapter 9](https://github.com/poare/parallel_lqcd/blob/main/Chapter-9---Wilson-Dslash-Kernel-From-Lattice_2015_High-Performance-Parallel.pdf)
- [High-Performance Lattice QCD for Multi-core Based Parallel Systems Using a Cache-Friendly Hybrid Threaded-MPI Approach](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6114421)

### What is implemented so far?
- src/actions/actions.jl: General framework for implementing actions.
  - We can use the Actions.FermiActionParam_Wilson
- src/fermions/AbstractFermion.jl
  - This is the abstract parent class of all fermions, and is worth taking a look at.
  - Two classes of interest: ```AbstractFermion```, and ```FermionFields```.
    - AbstractFermion is a superclass of all Fermion types, and we'll need it to make Wilson Fermions
    - Not sure about the implementation of FermionFields, but this is what ψ(x) will need to be.
- src/fermions/WilsonFermion.jl
  - This is a subtype ```WilsonFermion <: FermionFields```, so it's a field for a Wilson fermion.
  - I believe **either Wx!(...) or Dx!(...)** is the implementation of the Dirac operator: I'm not yet sure which one it is though.
  - Also a lot of extra functionality for multiplying Dirac fermions by γ matrices.
- src/gaugefields/gaugefields.jl
  - TODO check this out

### What do we need to implement?
- Extra data structures, like:
  - ```ParWilsonFermion <: WilsonFermion```: Our WilsonFermion module to use for this project, this is where we can add extra functionality that we want
  - ```HalfSpinor```: upper two components of a spinor field. Can also have this be absorbed into ```ParWilsonFermion```.
  - ```ParWilsonAction```: Multithreaded Wilson action.
  - ```MPIWilsonAction```: Distributed Wilson action using MPI. Will likely need separate run scripts as well.
- Blocking (?)

### Steps:
1. Implement the Wilson action in serial using LatticeQCD.jl datastructures.
2. Implement the HalfSpinor class to speed up the $\gamma$ matrix projections in the Wilson action.
3. Multithread (or use MPI) the Wilson action using blocking techniques to separate the lattice into sublattices.
4. Implement a conjugate gradient (CG) inverter. I'm not sure at the moment if this is something that we need to parallelize directly, or if we'd rather just call the parallelized Wilson kernel (i.e. the parallel ```Dx!``` code that we'll be writing)

### Random comments and notes
- To load a module, it's not as simple as just writing ```using LatticeQCD```. Let's say we want to use some functionality from the Actions module. We can either import it by selecting the functions we want to use, and using them, like so:
  ```
  using LatticeQCD.Actions:FermiActionParam_Wilson
  FermiActionParam_Wilson(...parameters...)
  ```
  or we can load the entire namespace ```Actions``` into our file, and use the keyword to access the contents of the module:
  ```
  using LatticeQCD.Actions:Actions
  Actions.FermiActionParam_Wilson(...parameters...)
  ```
- The notation ```Array{T,1}``` is just an array with 1 dimension, i.e. a ```Vector{T}```.
- A ```view``` of an Array gives a specific subset / slicing of the array, but allows the user to modify the original array by modifying the view. This is different than just slicing the array, i.e. ```A[2, :]```, because slicing the array gives a **copy** of the original array, while ```@view A[2, :]``` allows you to modify the original array by modifying the view object.
- To initialize julia with N threads, from the command line run:
  ```
  julia --nthreads N
  ```
- Git push and git pull: to pull just run ```git pull```. To push, you need to first commit, which you can do by specifying a message, for example the message "first commit":
  ```
  git commit -m "first commit"
  ```
  Then you should push:
  ```
  git push
  ```

### Running on wombat:
- Julia MPI scripts now run: Julia is built, and MPI.jl is built with the correct version of openmpi that is native to the cluster.
  The default submit script to copy is ```parallel```
- If that doesn't pan out, we can salloc, i.e. use
  ```
  salloc --time=00:30:00 --ntasks=4
  ```
  We can also request multiple nodes if we want, I believe the switch is -n.
  Once you salloc, you can run a script with the usual mpirun (make sure that
  openmpi is loaded, might need to do a ```module load openmpi```:
  ```
  poare@p55n3:~/parallel_lqcd/test/wombat$ mpirun -np 4 julia hello_world.jl
  ```

### Gauge field configurations
- LatticeQCD.jl has some gauge field configuration readers built in, in the ```/src/output/io.jl``` and ```/src/output/ildg_format.jl```. We'll
  be using the LIME data format, which you can check out in the ```ildg_format.jl``` file.
- There aren't a lot of possible gauge field configurations on the cluster which are pure Wilson action (most of the gauge fields are generated with more complicated actions). However, there's an ensemble of $16^3\times 48$ dimensional lattices with the Wilson action, which will be good ones to test our algorithms on (there are also larger Wilson action ensembles in the same parent directory). They're stored at
  ```
  /data/d10b/ensembles/quenched/su3_16_32_b5p87793/su3_16_32_b5p87793-stream1/cfgs/.
  ```
  I pulled a single gauge field configuration and put it in the directory ```parallel_lqcd/configs/su3_16_32_b5p87793/su3_16_32_b5p87793-stream1.lime99```. It's pretty large, but it'll be a good thing to have to make sure we can read data in. 
