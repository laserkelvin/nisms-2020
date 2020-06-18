using CSV, DataFrames, Plots, StatsPlots, Missings
using Flux
using Flux: relu
using Flux: mse, crossentropy
using Flux.Optimise: update!
using MLDataUtils
using StatsBase
using Random; Random.seed!(0)
using ColorSchemes
using Base.Iterators
using BSON: @save, @load

data = CSV.read("julia/cleaned.csv");

function df_to_data(dataframe)
    # Get the numeric data
    extracted = Matrix(dataframe[[:Lat, :Long, :PosEncoding, :HotDogEncoding]]);
    # calculate cartesian coordinates from lat/long
    x = cos.(extracted[:,1]) .* cos.(extracted[:,2])
    y = cos.(extracted[:,1]) .* sin.(extracted[:,2])
    z = sin.(extracted[:,1])

    X = hcat(x, y, z, extracted[:,3], dataframe[:,:RelativeTime]);
    dt = fit(UnitRangeTransform, X, dims=1);
    norm_X = StatsBase.transform(dt, X);
    norm_X = transpose(norm_X)
    Y = convert(Array{Float32}, extracted[:,4]);

    return X, Y
end

X, Y = df_to_data(data);

# Define a logistic regression model; output is sigmoid
embedder = Chain(
    Dense(5, 12, tanh),
    # Dense(16, 8, tanh),
    Dense(12, 2, tanh),
)
predictor = Dense(2, 1, σ)

dumber_model = Chain(embedder, predictor)

calc_loss(x) = mse(dumber_model(x), transpose(Y[:,:]))

dumber_model(norm_X)

# Setup training logistics
θ = Flux.params(dumber_model);
optimizer = NADAM(1e-3);

for i in range(1, stop=50000)
    grads = gradient(() -> calc_loss(norm_X), θ)
    for p in θ
        update!(optimizer, p, grads[p])
    end
    loss = calc_loss(norm_X)
    if mod(i, 1000) == 0
        println("Current loss is $loss")
    end
end

# Evaluate feature importance with input gradients; throw
# ones at the model and see how each affects the final predicted
# sigmoid value
durr = ones(5, 1)

gs = gradient(Flux.params(durr)) do
         sum(dumber_model(durr))
       end

# normalize by gradient magnitude
inp_grads = abs.(gs[durr]);
inp_grads ./= maximum(inp_grads);
# Make a plot of the input gradients
gr(size=(600, 500))
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
savefig("feature_importance.png")

# Visualizing the embeddings.

# The second to last layer is of 2D, with the idea
# that this corresponds to a 2D encoding of the "dataset"
# that we can use to evaluate a continuous decision boundary
embeddings = embedder(norm_X)
pred_Y = dumber_model(norm_X)

gr(size=(900, 500))

plot(
    scatter(embeddings[1,:], embeddings[2,:], zcolor=Y[:], leg=false, title="Truth"),
    scatter(embeddings[1,:], embeddings[2,:], zcolor=pred_Y[:], leg=false, title="Predicted")
    )
xlabel!("X")
ylabel!("Y")

scatter(embeddings[1,:], embeddings[2,:], zcolor=norm_X[4,:], leg=false)
xlabel!("X")
ylabel!("Y")

groups = groupby(data, :PosEncoding)

# See how the postdocs look
sliced_X, sliced_Y = df_to_data(groups[3])

slice_embeddings = embedder(transpose(sliced_X))


plot(
    scatter(slice_embeddings[1,:], slice_embeddings[2,:], zcolor=sliced_Y[:], leg=false, title="Truth", alpha=1.),
    # scatter(embeddings[1,:], embeddings[2,:], zcolor=pred_Y[:], leg=false, title="Predicted")
    )
scatter!(embeddings[1,:], embeddings[2,:], zcolor=Y[:], leg=false, alpha=0.1)
xlabel!("X")
ylabel!("Y")

function generate_grid(start, stop, length)
    linear_array = range(start, stop=stop, length=length)
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

grid_array = generate_grid(-1., 1., 100)

herp = range(-1., stop=1., length=100);
continuous = predictor(grid_array);

embeddings = embedder(transpose(X))
pred_Y = dumber_model(transpose(X))

contourf(herp, herp, continuous; levels = collect(range(0,stop=1,length=10)), c=:Spectral_6)
scatter!(
    embeddings[1,:], embeddings[2,:], zcolor=Y[:], lw=1.,
    markerstrokecolor="white", leg=false, c=:RdBu_10
    )
xlabel!("X")
ylabel!("Y")
title!("Wrong Decision Boundary")
savefig("decision_boundary.png")

@save "trained_model.bson" θ
