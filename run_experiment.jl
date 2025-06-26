# File: run_experiments.jl

include("MIPT_SimTools.jl")
using .MIPT_SimTools
using Dates

# ------------------------------------------------------------------
# 1. CHOOSE YOUR SIMULATOR
# ------------------------------------------------------------------
# Options: :MPS or :EXACT
const SIMULATOR_MODE = :MPS

# ------------------------------------------------------------------
# 2. DEFINE YOUR EXPERIMENT PARAMETERS
# ------------------------------------------------------------------
# The code will select the appropriate parameters based on the mode.
# Note: The exact simulator is much slower and memory-intensive!
# N > 14 can be very challenging.

const param_space = Dict(
    # --- Common Parameters ---
    :N           => [8],
    :l           => [80], # Example: Sweeping depth for the exact sim
    :p           => [0,0.2,0.4,0.6,0.8],
    :renyi_alpha => [2.0],         # 1.0 for von Neumann, 2.0 for Rényi-2

    # --- MPS-Specific Parameters (ignored in :EXACT mode) ---
    :maxdim      => [64],
    :cutoff      => [1e-8]
)

const num_trials = 100

# ------------------------------------------------------------------
# 3. RUN EXPERIMENT AND SAVE RESULTS
# ------------------------------------------------------------------
# Start Julia with `julia -t auto` for parallel execution.
println("Starting experiment: $(SIMULATOR_MODE) mode at $(now())")

if SIMULATOR_MODE == :MPS
    results_dataframe = run_parameter_sweep(param_space, num_trials)
elseif SIMULATOR_MODE == :EXACT
    results_dataframe = run_parameter_sweep_exact(param_space, num_trials)
else
    error("Invalid SIMULATOR_MODE. Choose :MPS or :EXACT.")
end

save_results(results_dataframe, param_space, SIMULATOR_MODE)

println("\nExperiment finished at $(now())")