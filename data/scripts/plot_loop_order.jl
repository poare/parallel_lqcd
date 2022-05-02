using HDF5
using Plots

file = "/Users/theoares/parallel_lqcd/data/loop_order.h5"
f = h5open(file, "r")
n_trials = 4
max_threads = 8
times = Array{Float64}(undef, n_trials, max_threads)
for tidx = 1 : max_threads
    times[:, tidx] = read(f["nt$(tidx)"])
end
close(f)

labels = ["serial" "th1" "th2" "th3"]
plot(1:max_threads, times', xlabel = "Threads", ylabel = "__rmul__ time", label = labels)
savefig("/Users/theoares/parallel_lqcd/data/plots/loop_order.pdf")
println("Done.")
