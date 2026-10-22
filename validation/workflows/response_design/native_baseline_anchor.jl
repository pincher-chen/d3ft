# Run the unchanged published Figure-1 implementation at original settings.
length(ARGS)==2 || error("upstream_repository output_prefix")
upstream,prefix=ARGS
mkpath(dirname(prefix))
include(joinpath(upstream,"figure1_derivative_combinations/lib.jl"))
DFTK.setup_threading()
run(prefix; Ecut=30,nkpt=4,displace=false)
