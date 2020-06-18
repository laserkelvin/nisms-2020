using CSV, DataFrames, Plots, StatsPlots, Missings
using Flux
using MLDataUtils
using PyCall

data = CSV.read("julia/cleaned.csv")

### Analyze the participants by career stage
career_group = groupby(data, :Position);
career_breakdown = sort(combine(career_group, nrow), :x1);
career_breakdown[!, :Index] = [1, 3, 4, 2];
sort!(career_breakdown, :Index);

career_breakdown.Position

# plot the data
gr(size=(600, 500))
chart = bar(
    career_breakdown[!, :x1],
    color = "#b85231",
    leg = false,
    lw = 1.0,
    bar_width = 0.3,
    background_color = "#f9efde",
    titlefont = font(14, "Open Sans"),
)
title!("Participant breakdown")
ylabel!("Counts", font = font(12, "Open Sans"))
xlabel!("Career stage", font = font(12, "Open Sans"))
xticks!([1, 2, 3, 4], career_breakdown[!, :Position])
savefig("career_plot.png")

# Generate a corner plot for everything
mat_rep = Matrix(data[[:Lat, :Long, :HotDogEncoding, :PosEncoding]]);

gr(size = (600, 500))
corrplot(
    mat_rep,
    label = ["Latitude", "Longitude", "Hot Dog?", "Career Step"],
    grid = false,
    compact = true,
)

# Get raw stats out; how likely each career stage is to be affirmative
true_stats = combine(career_group, :HotDogEncoding => sum);
true_stats[!,:nentries] = combine(career_group, nrow)[:x1]
true_stats[!,:percent] = true_stats[:HotDogEncoding_sum] ./ true_stats[:nentries];
true_stats[!,:Index] = [2,4,3,1];
sort!(true_stats, :Index)

gr(size=(600, 500))
chart = bar(
    true_stats[!, :percent],
    color = "#b85231",
    leg = false,
    lw = 1.0,
    bar_width = 0.3,
    background_color = "#f9efde",
    titlefont = font(14, "Open Sans"),
)
hline!([1.], lw=1.5)
title!("Sandwich vs. Career stage")
ylabel!("Fraction of Yes", font = font(12, "Open Sans"))
xlabel!("Career stage", font = font(12, "Open Sans"))
xticks!([1, 2, 3, 4], career_breakdown[!, :Position])
annotate!(1.3, 0.95, "Correct answer", font=font(1, "Open Sans"))
savefig("career_sandwich.png")
