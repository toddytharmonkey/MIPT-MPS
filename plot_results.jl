# File: plot_results.jl

using JLD2
using DataFrames
using Plots
using Printf

# ------------------------------------------------------------------
# 1. CONFIGURE YOUR PLOT
# ------------------------------------------------------------------
# --- The filename of the results you want to plot ---
filename = "MIPT_sweep_results_2025-06-25_14-00-00.jld2" # <-- UPDATE THIS

# --- Choose the independent variable for the x-axis ---
# Options: :p, :l, :N, :maxdim, etc. (must be a key in your param_space)
x_variable = :l

# --- Filter the data: Choose fixed values for all other parameters ---
# The script will find all data points that match these fixed values.
filters = (
    N = 12,
    p = 0.20,
    maxdim = 64,
    renyi_alpha = 2.0
)

# ------------------------------------------------------------------
# 2. LOAD AND FILTER DATA
# ------------------------------------------------------------------
data = load(filename)
df = data["df"]

# Apply all filters to create a data slice for plotting
filtered_df = df
for (key, val) in pairs(filters)
    # Ensure the filtering key is not the chosen x-variable
    if key != x_variable
        filtered_df = filter(row -> row[key] == val, filtered_df)
    end
end

if isempty(filtered_df)
    error("No data found for the specified filters. Please check your plot configuration.")
end

# Sort the data by the x-axis variable for correct plotting
sort!(filtered_df, x_variable)

# ------------------------------------------------------------------
# 3. CREATE AND SAVE PLOT
# ------------------------------------------------------------------
# Extract columns for plotting
x_values = filtered_df[!, x_variable]
y_values = filtered_df[!, :entropy_mean]
y_errors = filtered_df[!, :entropy_std] ./ sqrt(data["param_space"][:num_trials][1]) # Calculate SEM

# Generate a descriptive title and labels
plot_title = "MIPT Entropy vs. $(x_variable)"
filter_str = join(["$k=$v" for (k,v) in pairs(filters) if k != x_variable], ", ")
plot_subtitle = "($(filter_str))"

plot(
    x_values,
    y_values,
    yerror = y_errors,
    xlabel = string(x_variable),
    ylabel = "Avg. Entropy",
    title = plot_title,
    titlefontsize=12,
    plot_title=plot_subtitle, # Use plot_title for subtitle functionality
    plot_titlevspan=0.1,
    plot_titlefontsize=10,
    label = "Entropy (with Std. Error of Mean)",
    legend = :topright,
    lw = 2,
    marker = :circle,
)

# Save the plot
output_filename = replace(basename(filename), ".jld2" => "_$(x_variable)_plot.png")
savefig(output_filename)

println("Plot saved as: $(output_filename)")