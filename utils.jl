module Utils

using DelimitedFiles
using Dierckx
using Plots 
## File munging/management
data_dir = "data/"
# data exists for first 200 lcvs
# data exploration
function read_dat(filename::String)
    readdlm(data_dir*filename, ',')
end

function plot_before_after(fileNo::String)
    before = read_dat("lcv"*"old"*fileNo*".dat")
    after = read_dat("lcv"*"new"*fileNo*".dat")

    plot(before[:, 1], before[:, 2], seriestype = :scatter)
    plot!(after[:, 1], after[:, 2], seriestype = :scatter)
end

function read_params()
    readdlm(data_dir*"parameters.csv", ',')
end

function munge_training_data(index::Int64)
    training_params = read_params()
    # only want beta-factor and a/c axial ratio 
    return training_params[index, 1], training_params[index, 3]
end

function munge_expected_data(fileNo::String)
    before = read_dat("lcv"*"old"*fileNo*".dat")
    after = read_dat("lcv"*"new"*fileNo*".dat")
    return before, after
end

## scientific helpers 
function interpolate(start_range::Int64, end_range::Int64)
    xi = collect(100:50:4250)
    before_interpolated = zeros(size(xi)[1], 2, 100) # 100x2x200
    after_interpolated = zeros(size(xi)[1], 2, 100) # 100x2x200

    if start_range == 101
        factor = end_range - start_range + 1
    elseif start_range == 201
        factor = end_range - start_range + 101
    else
        factor = 0
    end
    
    for sample in start_range:end_range
        fileNo = lpad(sample, 3, "0")

        before = read_dat("lcv"*"old"*fileNo*".dat")
        after = read_dat("lcv"*"new"*fileNo*".dat")

        before_spl = Spline1D(before[:, 1], before[:, 2])
        after_spl = Spline1D(after[:, 1], after[:, 2])
      
        before_interpolated[:, :, sample - factor] = hcat(xi, before_spl(xi)) 
        after_interpolated[:, :, sample - factor] = hcat(xi, after_spl(xi))
    end

    return before_interpolated, after_interpolated
end


end
