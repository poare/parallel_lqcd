for ii in `seq 1 8`
do
  julia --threads ${ii} ../test_loop_order.jl
done
julia ../../data/scripts/plot_loop_order.jl
