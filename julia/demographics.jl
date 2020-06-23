using CSV, DataFrames, Plots, StatsPlots, Missings, CategoricalArrays
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
savefig("julia/career_plot.png")

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
savefig("julia/career_sandwich.png")

for col in [:Position, :Hotdog]
    categorical!(data, col);
end

gr(size=(600, 500))
@df data violin(:Position, :HotDogEncoding, alpha=0.7, background_color = "#f9efde", color="#b85231", marker=(0.2,"#b85231",stroke(0)), leg=false)
@df data dotplot!(:Position, :HotDogEncoding, marker=(:black,stroke(0)), leg=false, alpha=0.4)
yticks!([0., 1.], ["No", "Yes"])
ylabel!("Response")
savefig("julia/career_sandwich.png")

yaynay = combine(groupby(data, :HotDogEncoding), nrow);

gr(size=(600, 500))
chart = bar(
    yaynay[!, :x1],
    color = "#b85231",
    leg = false,
    lw = 1.0,
    bar_width = 0.3,
    background_color = "#f9efde",
    titlefont = font(14, "Open Sans"),
)
title!("Are hotdogs sandwiches?")
ylabel!("Counts", font = font(12, "Open Sans"))
xlabel!("Response", font = font(12, "Open Sans"))
xticks!([1, 2], ["Yes", "No"])
annotate!(1., 70., "Correct answer", font=font(1, "Open Sans"))
savefig("julia/hotdog_responses.png")

US = filter(row -> row.Long < -20., data)
EU = filter(row -> row.Long > -20., data)

US_yay = sum(US.HotDogEncoding) / nrow(US)
EU_yay = sum(EU.HotDogEncoding) / nrow(EU)

df = DataFrame([[US_yay, EU_yay]])

gr(size=(600, 500))
chart = bar(
    df[:,:x1],
    color = "#b85231",
    leg = false,
    lw = 1.0,
    bar_width = 0.3,
    background_color = "#f9efde",
    titlefont = font(14, "Open Sans"),
)
# title!("Europeans more likely to be right")
ylabel!("Fraction of Yes", font = font(12, "Open Sans"))
xlabel!("Side of the World", font = font(12, "Open Sans"))
xticks!([1, 2], ["Negative longitude", "Positive longitude"])
annotate!(1.3, 0.95, "Correct answer", font=font(1, "Open Sans"))
hline!([1.], lw=1.5)
savefig("julia/split_world.png")

length(unique(data.Org))
