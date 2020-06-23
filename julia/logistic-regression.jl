using CSV, DataFrames, Plots, StatsPlots, Missings
using Flux
using Flux: relu
using Flux: mse, crossentropy
using Flux.Optimise: update!
using Zygote
using MLDataUtils
using StatsBase
using Random;
Random.seed!(0);
using ColorSchemes
using Base.Iterators
using BSON: @save, @load

data = CSV.read("julia/cleaned.csv");

function df_to_data(dataframe)
    # Get the numeric data
    extracted = Matrix(dataframe[[:Lat, :Long, :PosEncoding, :HotDogEncoding]])
    # calculate cartesian coordinates from lat/long
    x = cos.(extracted[:, 1]) .* cos.(extracted[:, 2])
    y = cos.(extracted[:, 1]) .* sin.(extracted[:, 2])
    z = sin.(extracted[:, 1])

    X = hcat(x, y, z, extracted[:, 3], dataframe[:, :RelativeTime])
    dt = fit(UnitRangeTransform, X, dims = 1)
    norm_X = StatsBase.transform(dt, X)
    norm_X = transpose(norm_X)
    Y = convert(Array{Float32}, extracted[:, 4])

    return X, Y
end

X, Y = df_to_data(data);
X = transpose(X);

# Define a logistic regression model; output is sigmoid
embedder = Chain(
    Dense(5, 12, tanh),
    # Dense(16, 8, tanh),
    Dense(12, 2, tanh),
)
predictor = Dense(2, 1, σ)
# combine the models together
dumber_model = Chain(embedder, predictor)

# if isfile("julia/trained_model.bson") == false
calc_loss(x) = mse(dumber_model(x), transpose(Y[:, :]))

dumber_model(X)

# Setup training logistics
θ = Flux.params(dumber_model);
optimizer = NADAM(1e-3);

if isfile("julia/trained_model.bson")
    for i in range(1, stop = 50000)
        grads = gradient(() -> calc_loss(X), θ)
        for p in θ
            update!(optimizer, p, grads[p])
        end
        loss = calc_loss(X)
        if mod(i, 1000) == 0
            println("Current loss is $loss")
        end
    end
else
    @load "julia/trained_model.bson" θ
    Flux.loadparams!(dumber_model, θ)
end

# Evaluate feature importance with input gradients; throw
# noise at the model and see how each affects the final predicted
# sigmoid value
durr = rand(5, 1000)

gs = gradient(Flux.params(durr)) do
    sum(dumber_model(durr))
end

# normalize by gradient magnitude
inp_grads = abs.(gs[durr])
inp_grads = mean(inp_grads, dims=2)
inp_grads ./= maximum(inp_grads);
# Make a plot of the input gradients
gr(size = (600, 500))
chart = bar(
    inp_grads,
    color = "#b85231",
    leg = false,
    lw = 1.0,
    bar_width = 0.3,
    background_color = "#f9efde",
    titlefont = font(14, "Open Sans"),
)
title!("What decides what a sandwich?")
ylabel!("Importance", font = font(12, "Open Sans"))
xlabel!("Feature", font = font(12, "Open Sans"))
xticks!([1, 2, 3, 4, 5], ["X", "Y", "Z", "Career stage", "Registration time"])
savefig("julia/feature_importance.png")

# Visualizing the embeddings.

# The second to last layer is of 2D, with the idea
# that this corresponds to a 2D encoding of the "dataset"
# that we can use to evaluate a continuous decision boundary
embeddings = embedder(X)
pred_Y = dumber_model(X)

gr(size = (900, 500))

plot(
    scatter(
        embeddings[1, :],
        embeddings[2, :],
        zcolor = Y[:],
        leg = false,
        title = "Truth",
    ),
    scatter(
        embeddings[1, :],
        embeddings[2, :],
        zcolor = pred_Y[:],
        leg = false,
        title = "Predicted",
    ),
)
xlabel!("X")
ylabel!("Y")

groups = groupby(data, :PosEncoding)

# See how the postdocs look
sliced_X, sliced_Y = df_to_data(groups[3])

slice_embeddings = embedder(transpose(sliced_X))

gr(size=(600, 500))
plot(scatter(
    slice_embeddings[1, :],
    slice_embeddings[2, :],
    zcolor = sliced_Y[:],
    leg = false,
    title = "Truth",
    alpha = 1.0,
),
# scatter(embeddings[1,:], embeddings[2,:], zcolor=pred_Y[:], leg=false, title="Predicted")
)
scatter!(embeddings[1, :], embeddings[2, :], zcolor = Y[:], leg = false, alpha = 0.1)
xlabel!("X")
ylabel!("Y")

function generate_grid(start, stop, length)
    linear_array = range(start, stop = stop, length = length)
    grid = zeros(2, length * length)
    index = 1
    for a in linear_array
        for b in linear_array
            grid[1, index] = a
            grid[2, index] = b
            index = index + 1
        end
    end
    return grid
end

grid_array = generate_grid(-1.0, 1.0, 100)

herp = range(-1.0, stop = 1.0, length = 100);
continuous = predictor(grid_array);

embeddings = embedder(X)
pred_Y = dumber_model(X)

contourf(
    herp,
    herp,
    continuous;
    levels = collect(range(0, stop = 1, length = 10)),
    c = :Spectral_6,
    background_color = "#f9efde",
)
scatter!(
    embeddings[1, :],
    embeddings[2, :],
    zcolor = Y[:],
    lw = 1.0,
    markerstrokecolor = "white",
    leg = false,
    # c = :RdBu_10,
)
xlabel!("X")
ylabel!("Y")
title!("Wrong Decision Boundary")
savefig("julia/decision_boundary.png")

plots = [];
for (index, group) in enumerate(groupby(data, :PosEncoding))
    name = unique(group.Position)[1]
    temp_X, temp_Y = df_to_data(group)
    temp_embedding = embedder(transpose(temp_X))
    temp_plot = contourf(
        herp,
        herp,
        continuous;
        levels = collect(range(0, stop = 1, length = 10)),
        c = :Spectral_6,
        background_color = "#f9efde",
        alpha=0.8
    )
    scatter!(
        temp_embedding[1, :],
        temp_embedding[2, :],
        zcolor = temp_Y[:],
        lw = 1.0,
        markerstrokecolor = "white",
        leg = false
        # c = :RdBu_10,
    )
    scatter!(
        embeddings[1, :],
        embeddings[2, :],
        zcolor = Y[:],
        leg = false,
        alpha=0.1
    ),
    title!(name)
    push!(plots, temp_plot)
    savefig("julia/$name-contour.png")
end

plots[4]
# dump the weights to disk
# @save "julia/trained_model.bson" θ

sum(data.HotDogEncoding)
