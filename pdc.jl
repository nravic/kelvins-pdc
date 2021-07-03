using DifferentialEquations
using Flux
using Flux: train!
using Statistics
using Zygote
using BSON: @save
include("utils.jl")


struct TrainingParams
    epochs::Int64
    learning_rate::Float64
end

# Straight shot ML, how bad could it be?
# Some notes - temporality not taken into account, maybe try an RNN for that?
function construct_model()
    model = Chain(
        Dense(84, 21, relu),
        Dense(21, 7, relu), 
        Dense(7, 2))
end

function train_barebones!(hyper_params::TrainingParams, model)
    # given a before and after lightcurves. Can we encode data in the difference b/ween them?
    before, after = Utils.interpolate(1, 100)
    
    opt = Descent(hyper_params.learning_rate)
    loss(x, y) = Flux.Losses.mse(model(x), y)
    pm = params(model)
    epochs = hyper_params.epochs
    
    epoch_loss = zeros(epochs)
    for epoch in 1:epochs
        interm_loss = zeros(100)
        
        for sample in 1:100
            y_train = Utils.munge_training_data(sample)
            x_train = before[:, 2, sample] - after[:, 2, sample]
            data = [(x_train, collect(y_train))]
            
            train!(loss, pm, data, opt)
            
            interm_loss[sample] = loss(x_train, y_train)
            @show(interm_loss[sample])
        end

        @show(params(model))
        epoch_loss[epoch] = mean(interm_loss)
    end

    weights = params(model)

    @save "barebones.bson" weights 
end

function validate(model)
    # validate model w/ test data
    before_val, after_val = Utils.interpolate(101, 200)
    loss_validation = zeros(size(before_val)[3])
    loss(x, y) = Flux.Losses.mse(model(x), y)

    for sample in 1:size(before_val)[3]
        y_val = Utils.munge_training_data(100 + sample)
        x_val = before_val[:, 2, sample] - after_val[:, 2, sample]
        loss_validation[sample] = loss(x_val, y_val)
    end

    return loss_validation
end

function generate_final(model)
    before, after = Utils.interpolate(201, 300)
    loss(x, y) = Flux.Losses.mse(model(x), y)
    
    final_loss = zeros(100)
    final_vals = zeros(100, 2)
    for sample in 1:100
        x_final = before[:, 2, sample] - after[:, 2, sample]
        y_final = Utils.munge_training_data(sample)

        final_vals[sample, :] = model(x_final)
        final_loss[sample] = loss(x_final, y_final)
    end
    # annoyingly they also want the J value... so
    j_val = zeros(100)
    final_sub = hcat(final_vals[:, 1], j_val, final_vals[:, 2])
    return final_loss, final_sub
end

        
