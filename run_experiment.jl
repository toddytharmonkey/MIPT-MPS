# File: run_experiments.jl

# --- Include the simulation engine and its functions ---
include("MIPT_SimTools.jl")
using .MIPT_SimTools

# ------------------------------------------------------------------
# 1. DEFINE YOUR EXPERIMENT
# ------------------------------------------------------------------
# Define the lists of parameters you want to sweep over.
# The code will run all combinations of these parameters.
# To fix a parameter, just provide a single-element list e.g., `:N => [12]`.

const param_space = Dict(
    :N           => [12],
    :l           => [24, 48, 72, 96, 120, 160],
    :p           => [0.20], # Fixed p for a layer sweep
    :maxdim      => [64],
    :renyi_alpha => [2.0]     # 2.0 for Rényi-2, 1.0 for von Neumann
)

# Example for a probability sweep:
# const param_space = Dict(
#     :N           => [12, 16],
#     :l           => [120],
#     :p           => 0.0:0.02:0.3,
#     :maxdim      => [64],
#     :renyi_alpha => [2.0]
# )

const num_trials = 100

# ------------------------------------------------------------------
# 2. RUN EXPERIMENT AND SAVE RESULTS
# ------------------------------------------------------------------
# Start Julia with `julia -t auto` for parallel execution.
println("Starting experiment at $(now())")

results_dataframe = run_parameter_sweep(param_space, num_trials)
save_results(results_dataframe, param_space)

println("\nExperiment finished at $(now())")