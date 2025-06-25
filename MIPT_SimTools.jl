module MIPT_SimTools

using ITensors
using ITensorMPS
using ITensorEntropyTools
using LinearAlgebra
using Random
using Dates
using JLD2
using Printf
using ProgressMeter
using DataFrames 
using IterTools 

export run_parameter_sweep, run_parameter_sweep_exact, save_results


# ==================================================================
# SECTION 2: MPS SIMULATOR 
# ==================================================================

# --- The function for a SINGLE trial, now takes a NamedTuple of parameters ---
function run_single_trial(params)
    # Unpack parameters
    N, l, p, maxdim, cutoff, renyi_alpha = params.N, params.l, params.p, params.maxdim, params.cutoff, params.renyi_alpha
    
    sites = siteinds("Qubit", N)
    psi = MPS(sites, "0")

    for _ in 1:l
        # Unitary layers with normalization
        for j in 1:2:N-1
            gate = random_unitary_gate(sites[j], sites[j+1])
            psi = apply(gate, psi; cutoff, maxdim)
            normalize!(psi)
        end
        for j in 2:2:N-1
            gate = random_unitary_gate(sites[j], sites[j+1])
            psi = apply(gate, psi; cutoff, maxdim)
            normalize!(psi)
        end
        
        # Measurement layer
        ops = OpSum()
        measured_something = false
        for j in 1:N
            if rand() < p
                measured_something = true
                prob_1 = expect(psi, "Proj1"; sites=j)[1]
                outcome = rand() < prob_1 ? "Proj1" : "Proj0"
                add!(ops, outcome, j)
            end
        end
        if measured_something
            meas_mpo = MPO(ops, sites)
            psi = apply(meas_mpo, psi; cutoff, maxdim)
            normalize!(psi)
        end
    end

    # Entropy calculation
    bipartition_site = N ÷ 2
    region = 1:bipartition_site
    Sn = ee_region(psi, collect(region); ee_type=EEType("Renyi"), n=renyi_alpha)
    return Sn
end

# --- The main runner function that handles parameter sweeps and parallelization ---
function run_parameter_sweep(param_space::Dict, num_trials::Int)
    # Generate all unique combinations of parameters
    param_combinations = collect(IterTools.product(
        get(param_space, :N, [12]),
        get(param_space, :l, [120]),
        get(param_space, :p, [0.2]),
        get(param_space, :maxdim, [64]),
        get(param_space, :cutoff, [1e-8]),
        get(param_space, :renyi_alpha, [2.0])
    ))

    num_setups = length(param_combinations)
    println("Starting parameter sweep with $(num_setups) unique setups and $(num_trials) trials each.")
    @printf("Total simulations to run: %d\n\n", num_setups * num_trials)

    # Prepare a DataFrame to store all results
    results_df = DataFrame(
        N=Int[], l=Int[], p=Float64[], maxdim=Int[], renyi_alpha=Float64[],
        entropy_mean=Float64[], entropy_std=Float64[]
    )

    @showprogress "Sweeping Parameters..." for (i, param_tuple) in enumerate(param_combinations)
        params = (N=param_tuple[1], l=param_tuple[2], p=param_tuple[3], maxdim=param_tuple[4],
                  cutoff=param_tuple[5], renyi_alpha=param_tuple[6])
        
        trial_entropies = Vector{Float64}(undef, num_trials)

        # We can still parallelize the trials for each parameter set
        Threads.@threads for trial in 1:num_trials
            trial_entropies[trial] = run_single_trial(params)
        end

        # Calculate statistics and add a row to the DataFrame
        mean_S = sum(trial_entropies) / num_trials
        std_S = num_trials > 1 ? sqrt(sum((trial_entropies .- mean_S).^2) / (num_trials - 1)) : 0.0
        
        push!(results_df, (params.N, params.l, params.p, params.maxdim, params.renyi_alpha, mean_S, std_S))
    end

    return results_df
end


# ==================================================================
# SECTION 2: EXACT (FULL STATE VECTOR) SIMULATOR
# ==================================================================

function run_single_trial_exact(params)
    # Unpack parameters - note there is no `maxdim` or `cutoff`
    N, l, p, renyi_alpha = params.N, params.l, params.p, params.renyi_alpha
    
    indices = siteinds("Qubit", N)
    state = onehot(ComplexF64, indices...)

    for _ in 1:l
        # Unitary layers
        start_site = l % 2 == 1 ? 1 : 2
        for j = start_site:2:(N-1)
            U_tensor = random_unitary_gate(indices[j], indices[j+1])
            state = apply(U_tensor, state)
            state /= norm(state)
        end
        
        # Measurement layer
        for j = 1:N
            if rand() < p
                site_index = indices[j]
                P0 = op("Proj0", site_index)
                
                # Compute probability of measuring 0
                ψ_temp = apply(P0, state)
                p0 = real(inner(ψ_temp, ψ_temp))
                
                # Decide which projector to apply based on Born rule
                Proj_tensor = rand() < p0 ? op("Proj0", site_index) : op("Proj1", site_index)
                
                state = apply(Proj_tensor, state)
                state /= norm(state) # Normalize after projection
            end
        end
    end

    # Entropy calculation using the same consistent tool
    bipartition_site = N ÷ 2
    region = indices[1:bipartition_site]
    return renyi_entropy(state, region, renyi_alpha)
end

function run_parameter_sweep_exact(param_space::Dict, num_trials::Int)
    # Generate parameter combinations, ignoring MPS-specific ones like :maxdim
    param_combinations = collect(IterTools.product(
        get(param_space, :N, [10]),
        get(param_space, :l, [100]),
        get(param_space, :p, [0.2]),
        get(param_space, :renyi_alpha, [2.0])
    ))
    num_setups = length(param_combinations)
    println("Starting EXACT parameter sweep with $(num_setups) unique setups and $(num_trials) trials each.")

    # Modified DataFrame for exact results (no maxdim column)
    results_df = DataFrame(N=Int[], l=Int[], p=Float64[], renyi_alpha=Float64[], entropy_mean=Float64[], entropy_std=Float64[])
    
    @showprogress "Sweeping Parameters (Exact)..." for param_tuple in param_combinations
        params = (N=param_tuple[1], l=param_tuple[2], p=param_tuple[3], renyi_alpha=param_tuple[4])
        
        trial_entropies = Vector{Float64}(undef, num_trials)
        # Using the same parallel trial structure
        Threads.@threads for trial in 1:num_trials
            trial_entropies[trial] = run_single_trial_exact(params)
        end

        mean_S = sum(trial_entropies) / num_trials
        std_S = num_trials > 1 ? sqrt(sum((trial_entropies .- mean_S).^2) / (num_trials - 1)) : 0.0
        push!(results_df, (params.N, params.l, params.p, params.renyi_alpha, mean_S, std_S))
    end
    return results_df
end

# ==================================================================
# SECTION 3: HELPER FUNCTIONS 
# ==================================================================

# --- Helper to create the random unitary gate ---
function random_unitary_gate(s1::Index, s2::Index)
    dim = ITensors.dim(s1) * ITensors.dim(s2)
    M = randn(ComplexF64, dim, dim)
    Q, _ = qr(M)
    return ITensor(Matrix(Q), s2', s1', s2, s1)
end

# --- Data saving function ---
function save_results(df::DataFrame, param_space::Dict)
    timestamp = Dates.format(now(), "YYYY-mm-dd_HH-MM-SS")
    filename = "MIPT_sweep_results_$(timestamp).jld2"
    println("\nSaving results to JLD2 file: $(filename)")
    
    # Save the DataFrame and the parameter space dictionary that generated it
    jldsave(filename; df=df, param_space=param_space)
    println("File saved.")
    return filename
end
# Renyi entropy for a full ITensor state vector
function renyi_entropy(ψ::ITensor, region::Vector{<:Index}, α::Real)
    # Orthogonalize the state with respect to the region
    # This performs an SVD on the bipartition
    U, S, V = svd(ψ, region)
    
    # The diagonal elements of S are the singular values
    # The squared singular values are the eigenvalues of the reduced density matrix
    p = diag(array(S, commoninds(S,V), commoninds(S,U))) .^ 2
    
    # Filter out zero probabilities for numerical stability with log
    p = p[p .> 1e-12]

    if α == 1.0 # Von Neumann entropy
        return -sum(p .* log.(p))
    else # Rényi-α entropy
        return (1 / (1 - α)) * log(sum(p .^ α))
    end
end

end # end module