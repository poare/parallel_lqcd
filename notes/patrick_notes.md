# Note document for Patrick

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

### Benchmarking __rmul!__
- The way we multithread these operators from LatticeQCD.jl is interesting to look at. The natural way to multithread is over the Dirac spinor index $\alpha$, since that's the first component of the array. However, we may want to think about changing that. For multithreading up $N_d$ (the maximum value of $\alpha$), this works fine, but $N_d$ will be 2 (for two-spinors) or 4 (for full Dirac spinors), so either when we hit 2 or 4 threads we'll achieve maximum speedup, regardless of if the number of threads is 4, 6, or 8.
- Instead, the best way to multithread may be looping over $t$, or the first position component after $\alpha$, since the temporal size of the lattice will almost always be $\geq 8$.
  - Make a plot of these two situtations after different amounts of multithreading, for $n_\mathrm{threads} \in \{1, 2, 3, ..., 8\}$, to see the maximally efficient way to do this.
- Consider swapping the order of indices in the array: see if it's worth changing the way that (w::ParWilsonFermion).f is structured to take advantage of the `@simd` operation using the fact that Julia is column major
  - Note that if we do this, we likely would only need to change the backend, i.e. change how __get_index__ is implemented and we shouldn't need to change much other than that

### Parallelizing the Wilson-Dirac operator
- The idea here is to partition the lattice into different "blocks" to run on different cores. Each core will evaluate Dx! on each block, then we'll gather the results to get the application of Dx! on a fermion field. Typically, the blocks are partitioned in the X and T directions, and take up the entire Y and Z domains.
- We'll want to use SIMD in one dimension on one of the loops-- conventionally this is done in the $X$ direction.
- We can use multi-threading as well as blocking. On each block (which is on a separate processor), we can use $N_{thr}$ threads. Each thread can traverse different lattice sites. There are a few way to assign these specifications:
  1. Blocking the $Y$ direction: here thread 0 evalutes Dx! for Y = 0, thread 1 evaluates Dx! for the Y = 1 slice, .... Once you hit $N_{thr}$, just start over, so thread $i$ evaluates the Wilson operator at $Y \in \{i + k * N_{thr} : k\in\mathbb Z\}$.
  2. A $\sqrt{N_{thr}} \times \sqrt{N_{thr}}$ block.
  3. Scanning along $Y$ for different $Z$ coordinates.
-
