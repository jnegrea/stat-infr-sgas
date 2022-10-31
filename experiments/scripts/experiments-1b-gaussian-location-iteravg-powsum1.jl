using LinearAlgebra

source_path = "../core/"
include(source_path*"sgld-module-0.jl")
# using .SGLDModule

output_path = "../output/"
isdir(output_path) || mkdir(output_path)
## experiment-1 Gausian Location Experiments

output_path_1b = output_path*"experiments-1b-gaussianLocation/"
isdir(output_path_1b) || mkdir(output_path_1b)

## Example 1.1: Well-spec gaussian location -- a simple test case to make sure everything works
experiment_name = "experiments-1b-1-gaussianLocation-wellspec"
output_path_1b_1 = output_path_1b*experiment_name*"/"
isdir(output_path_1b_1) || mkdir(output_path_1b_1)

parmdim = 10
Σ = Matrix(Diagonal(ones(parmdim)))
Σlik = Matrix(Diagonal(ones(parmdim)))


samplesize = 100


iteravg_numepochs=8 # NEW
num_iteravgs = 10000
numepochs = num_iteravgs*iteravg_numepochs

w=1/2
pows = [1,2/3,1/2,1/3] # Not actually used, just a reference for JN & HF
algos = [:vanillaSGDPowOneSumOne, :vanillaSGDPowTwoThirdSumOne, :vanillaSGDPowHalfSumOne, :vanillaSGDPowThirdSumOne]
batchsizes = Dict(
    :vanillaSGDPowOneSumOne => 1*ceil(Int,samplesize^(1-1)),
    :vanillaSGDPowTwoThirdSumOne => 1*ceil(Int,samplesize^(1-2/3)),
    :vanillaSGDPowHalfSumOne => 1*ceil(Int,samplesize^(1-1/2)),
    :vanillaSGDPowThirdSumOne => 1*ceil(Int,samplesize^(1-1/3))
)
numstepss = Dict(
    algo => numepochs*samplesize ÷ batchsizes[algo] for algo in algos
)

layers = layerer([:sandwichScaled, :vanillaSGDPowOneSumOne, :iteravg, :trueparm])
layers = layerer(layers, :vanillaSGDPowTwoThirdSumOne, drop=:vanillaSGDPowOneSumOne)
layers = layerer(layers, :vanillaSGDPowHalfSumOne, drop=:vanillaSGDPowTwoThirdSumOne)
layers = layerer(layers, :vanillaSGDPowThirdSumOne, drop=:vanillaSGDPowHalfSumOne)


# gaussianLocationInit(samplesize, parmdim, Σ, Σlik, output_path_1b_1*"model_init.jld2")
# modelRun(output_path_1b_1*"model_init.jld2", output_path_1b_1*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true)
# modelPlot(output_path_1b_1*"model_run.jld2", output_path_1b_1*experiment_name*"-fig",
#     layers, (-3,3), (-3,3), nameattr= false, iteravg_numepochs=iteravg_numepochs)

## Example 1.2: well-spec multivariate gaussian location -- non-spherical
experiment_name = "experiments-1b-2-gaussianLocation-wellspec-nonspherical"
output_path_1b_2 = output_path_1b*experiment_name*"/"
isdir(output_path_1b_2) || mkdir(output_path_1b_2)

parmdim = 10
Σ = Matrix(Diagonal(1.0 * (1.0:(parmdim))))
Σlik = Matrix(Diagonal(1.0 * (1.0:(parmdim))))


samplesize = 100


iteravg_numepochs=8 # NEW
num_iteravgs = 10000
numepochs = num_iteravgs*iteravg_numepochs

w=1/2
pows = [1,2/3,1/2,1/3] # Not actually used, just a reference for JN & HF
algos = [:vanillaSGDPowOneSumOne, :vanillaSGDPowTwoThirdSumOne, :vanillaSGDPowHalfSumOne, :vanillaSGDPowThirdSumOne]
batchsizes = Dict(
    :vanillaSGDPowOneSumOne => 1*ceil(Int,samplesize^(1-1)),
    :vanillaSGDPowTwoThirdSumOne => 1*ceil(Int,samplesize^(1-2/3)),
    :vanillaSGDPowHalfSumOne => 1*ceil(Int,samplesize^(1-1/2)),
    :vanillaSGDPowThirdSumOne => 1*ceil(Int,samplesize^(1-1/3))
)
numstepss = Dict(
    algo => numepochs*samplesize ÷ batchsizes[algo] for algo in algos
)

layers = layerer([:sandwichScaled, :vanillaSGDPowOneSumOne, :iteravg, :trueparm])
layers = layerer(layers, :vanillaSGDPowTwoThirdSumOne, drop=:vanillaSGDPowOneSumOne)
layers = layerer(layers, :vanillaSGDPowHalfSumOne, drop=:vanillaSGDPowTwoThirdSumOne)
layers = layerer(layers, :vanillaSGDPowThirdSumOne, drop=:vanillaSGDPowHalfSumOne)


# gaussianLocationInit(samplesize, parmdim, Σ, Σlik, output_path_1b_2*"model_init.jld2")
# modelRun(output_path_1b_2*"model_init.jld2", output_path_1b_2*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true)
# modelPlot(output_path_1b_2*"model_run.jld2", output_path_1b_2*experiment_name*"-fig",
#     layers, (-3,3), (-3,3), nameattr= false, iteravg_numepochs=iteravg_numepochs)

## Example 1.3: miss-spec multivariate gaussian location with wrong covariance
experiment_name = "experiments-1b-3-gaussianLocation-missspec"
output_path_1b_3 = output_path_1b*experiment_name*"/"
isdir(output_path_1b_3) || mkdir(output_path_1b_3)

parmdim = 10
Σ = Diagonal(0.5*ones(parmdim)).+0.5
Σlik = Matrix(Diagonal(1:parmdim))


samplesize = 100


iteravg_numepochs=8 # NEW
num_iteravgs = 10000
numepochs = num_iteravgs*iteravg_numepochs

w=1/2
pows = [1,2/3,1/2,1/3] # Not actually used, just a reference for JN & HF
algos = [:vanillaSGDPowOneSumOne, :vanillaSGDPowTwoThirdSumOne, :vanillaSGDPowHalfSumOne, :vanillaSGDPowThirdSumOne]
batchsizes = Dict(
    :vanillaSGDPowOneSumOne => 1*ceil(Int,samplesize^(1-1)),
    :vanillaSGDPowTwoThirdSumOne => 1*ceil(Int,samplesize^(1-2/3)),
    :vanillaSGDPowHalfSumOne => 1*ceil(Int,samplesize^(1-1/2)),
    :vanillaSGDPowThirdSumOne => 1*ceil(Int,samplesize^(1-1/3))
)
numstepss = Dict(
    algo => numepochs*samplesize ÷ batchsizes[algo] for algo in algos
)

layers = layerer([:sandwichScaled, :vanillaSGDPowOneSumOne, :iteravg, :trueparm])
layers = layerer(layers, :vanillaSGDPowTwoThirdSumOne, drop=:vanillaSGDPowOneSumOne)
layers = layerer(layers, :vanillaSGDPowHalfSumOne, drop=:vanillaSGDPowTwoThirdSumOne)
layers = layerer(layers, :vanillaSGDPowThirdSumOne, drop=:vanillaSGDPowHalfSumOne)

#
# gaussianLocationInit(samplesize, parmdim, Σ, Σlik, output_path_1b_3*"model_init.jld2")
# modelRun(output_path_1b_3*"model_init.jld2", output_path_1b_3*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true)
# modelPlot(output_path_1b_3*"model_run.jld2", output_path_1b_3*experiment_name*"-fig",
#     layers, (-3,3), (-3,3), nameattr= false, iteravg_numepochs=iteravg_numepochs)

## Example 1.4: miss-spec multivariate gaussian location with wrong covariance, non-diagonal
experiment_name = "experiments-1b-4-gaussianLocation-missspec-non-diagonal"
output_path_1b_4 = output_path_1b*experiment_name*"/"
isdir(output_path_1b_4) || mkdir(output_path_1b_4)

parmdim = 10
Σ = Matrix(Diagonal(ones(parmdim)))
Σlik = Matrix(Diagonal(1.0 * (1.0:(parmdim))))


samplesize = 100


iteravg_numepochs=8 # NEW
num_iteravgs = 10000
numepochs = num_iteravgs*iteravg_numepochs

w=1/2
pows = [1,2/3,1/2,1/3] # Not actually used, just a reference for JN & HF
algos = [:vanillaSGDPowOneSumOne, :vanillaSGDPowTwoThirdSumOne, :vanillaSGDPowHalfSumOne, :vanillaSGDPowThirdSumOne]
batchsizes = Dict(
    :vanillaSGDPowOneSumOne => 1*ceil(Int,samplesize^(1-1)),
    :vanillaSGDPowTwoThirdSumOne => 1*ceil(Int,samplesize^(1-2/3)),
    :vanillaSGDPowHalfSumOne => 1*ceil(Int,samplesize^(1-1/2)),
    :vanillaSGDPowThirdSumOne => 1*ceil(Int,samplesize^(1-1/3))
)
numstepss = Dict(
    algo => numepochs*samplesize ÷ batchsizes[algo] for algo in algos
)

layers = layerer([:sandwichScaled, :vanillaSGDPowOneSumOne, :iteravg, :trueparm])
layers = layerer(layers, :vanillaSGDPowTwoThirdSumOne, drop=:vanillaSGDPowOneSumOne)
layers = layerer(layers, :vanillaSGDPowHalfSumOne, drop=:vanillaSGDPowTwoThirdSumOne)
layers = layerer(layers, :vanillaSGDPowThirdSumOne, drop=:vanillaSGDPowHalfSumOne)


# gaussianLocationInit(samplesize, parmdim, Σ, Σlik, output_path_1b_4*"model_init.jld2")
# modelRun(output_path_1b_4*"model_init.jld2", output_path_1b_4*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true)
# modelPlot(output_path_1b_4*"model_run.jld2", output_path_1b_4*experiment_name*"-fig",
#     layers, (-3,3), (-3,3), nameattr= false, iteravg_numepochs=iteravg_numepochs)

## Example 1.5: well-spec multivariate gaussian location with non-diagonal covariance
experiment_name = "experiments-1b-5-gaussianLocation-missspec"
output_path_1b_5 = output_path_1b*experiment_name*"/"
isdir(output_path_1b_5) || mkdir(output_path_1b_5)

parmdim = 10
Σ = 5*(Diagonal(0.1*ones(parmdim)).+0.9)
Σlik = Σ


samplesize = 100

iteravg_numepochs=8 # NEW
num_iteravgs = 10000
numepochs = num_iteravgs*iteravg_numepochs

w=1/2
pows = [1,2/3,1/2,1/3] # Not actually used, just a reference for JN & HF
algos = [:vanillaSGDPowOneSumOne, :vanillaSGDPowTwoThirdSumOne, :vanillaSGDPowHalfSumOne, :vanillaSGDPowThirdSumOne]
batchsizes = Dict(
    :vanillaSGDPowOneSumOne => 1*ceil(Int,samplesize^(1-1)),
    :vanillaSGDPowTwoThirdSumOne => 1*ceil(Int,samplesize^(1-2/3)),
    :vanillaSGDPowHalfSumOne => 1*ceil(Int,samplesize^(1-1/2)),
    :vanillaSGDPowThirdSumOne => 1*ceil(Int,samplesize^(1-1/3))
)
numstepss = Dict(
    algo => numepochs*samplesize ÷ batchsizes[algo] for algo in algos
)

layers = layerer([:sandwichScaled, :vanillaSGDPowOneSumOne, :iteravg, :trueparm])
layers = layerer(layers, :vanillaSGDPowTwoThirdSumOne, drop=:vanillaSGDPowOneSumOne)
layers = layerer(layers, :vanillaSGDPowHalfSumOne, drop=:vanillaSGDPowTwoThirdSumOne)
layers = layerer(layers, :vanillaSGDPowThirdSumOne, drop=:vanillaSGDPowHalfSumOne)


# gaussianLocationInit(samplesize, parmdim, Σ, Σlik, output_path_1b_5*"model_init.jld2")
# modelRun(output_path_1b_5*"model_init.jld2", output_path_1b_5*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true)
# modelPlot(output_path_1b_5*"model_run.jld2", output_path_1b_5*experiment_name*"-fig",
#     layers, (-3,3), (-3,3), nameattr= false, iteravg_numepochs=iteravg_numepochs)
## Example 1.6: well-spec multivariate gaussian location with non-diagonal covariance for presentation
experiment_name = "experiments-1b-6-gaussianLocation-missspec-presentation"
output_path_1b_6 = output_path_1b*experiment_name*"/"
isdir(output_path_1b_6) || mkdir(output_path_1b_6)

parmdim = 10
s=0.1
Σlik = 0.125*(Diagonal(ones(parmdim)))
Σ = (Diagonal(s*ones(parmdim)).+(1-s))



samplesize = 1000

iteravg_numepochs=8 # NEW
num_iteravgs = 1000
numepochs = num_iteravgs*iteravg_numepochs

w=1/2
pows = [1,2/3,1/2,1/3] # Not actually used, just a reference for JN & HF
algos = [:vanillaSGDPowOneSumOne, :vanillaSGDPowTwoThirdSumOne, :vanillaSGDPowHalfSumOne, :vanillaSGDPowThirdSumOne]
batchsizes = Dict(
    :vanillaSGDPowOneSumOne => 1*ceil(Int,samplesize^(1-1)),
    :vanillaSGDPowTwoThirdSumOne => 1*ceil(Int,samplesize^(1-2/3)),
    :vanillaSGDPowHalfSumOne => 1*ceil(Int,samplesize^(1-1/2)),
    :vanillaSGDPowThirdSumOne => 1*ceil(Int,samplesize^(1-1/3))
)
numstepss = Dict(
    algo => numepochs*samplesize ÷ batchsizes[algo] for algo in algos
)

layers = layerer([:sandwichScaled, :vanillaSGDPowOneSumOne, :iteravg, :trueparm])
layers = layerer(layers, :vanillaSGDPowTwoThirdSumOne, drop=:vanillaSGDPowOneSumOne)
layers = layerer(layers, :vanillaSGDPowHalfSumOne, drop=:vanillaSGDPowTwoThirdSumOne)
layers = layerer(layers, :vanillaSGDPowThirdSumOne, drop=:vanillaSGDPowHalfSumOne)


gaussianLocationInit(samplesize, parmdim, Σ, Σlik, output_path_1b_6*"model_init.jld2")
modelRun(output_path_1b_6*"model_init.jld2", output_path_1b_6*"model_run.jld2",
    algos, samplesize, batchsizes, w, numstepss; multithread=true)
modelPlot(output_path_1b_6*"model_run.jld2", output_path_1b_6*experiment_name*"-fig",
    layers, (-3,3), (-3,3), nameattr= false, iteravg_numepochs=iteravg_numepochs)


## Example 1.7: well-spec multivariate gaussian location with non-diagonal covariance for presentation
experiment_name = "experiments-1b-7-gaussianLocation-missspec-presentation-jsm"
output_path_1b_7 = output_path_1b*experiment_name*"/"
isdir(output_path_1b_7) || mkdir(output_path_1b_7)

parmdim = 10
s=0.1
Σlik = 0.125*(Diagonal(ones(parmdim)))
Σ = (Diagonal(s*ones(parmdim)).+(1-s))




samplesize = 1000

iteravg_numepochs=8 # NEW
num_iteravgs = 10000
numepochs = num_iteravgs*iteravg_numepochs

w=1/2
pows = [1,2/3,1/2,1/3] # Not actually used, just a reference for JN & HF
algos = [:vanillaSGDPowOneSumOne, :vanillaSGDPowTwoThirdSumOne, :vanillaSGDPowHalfSumOne, :vanillaSGDPowThirdSumOne]
batchsizes = Dict(
    :vanillaSGDPowOneSumOne => 1*ceil(Int,samplesize^(1-1)),
    :vanillaSGDPowTwoThirdSumOne => 1*ceil(Int,samplesize^(1-2/3)),
    :vanillaSGDPowHalfSumOne => 1*ceil(Int,samplesize^(1-1/2)),
    :vanillaSGDPowThirdSumOne => 1*ceil(Int,samplesize^(1-1/3))
)
numstepss = Dict(
    algo => numepochs*samplesize ÷ batchsizes[algo] for algo in algos
)

layers = layerer([:sandwichScaled, :vanillaSGDPowOneSumOne, :iteravg, :trueparm])
layers = layerer(layers, :vanillaSGDPowTwoThirdSumOne, drop=:vanillaSGDPowOneSumOne)
layers = layerer(layers, :vanillaSGDPowHalfSumOne, drop=:vanillaSGDPowTwoThirdSumOne)
layers = layerer(layers, :vanillaSGDPowThirdSumOne, drop=:vanillaSGDPowHalfSumOne)


gaussianLocationInit(samplesize, parmdim, Σ, Σlik, output_path_1b_7*"model_init.jld2")
modelRun(output_path_1b_7*"model_init.jld2", output_path_1b_7*"model_run.jld2",
    algos, samplesize, batchsizes, w, numstepss; multithread=true)
modelPlot(output_path_1b_7*"model_run.jld2", output_path_1b_7*experiment_name*"-fig",
    layers, (-3,3), (-3,3), nameattr= false, iteravg_numepochs=iteravg_numepochs)

## Example 1.8: miss-spec multivariate gaussian location with wrong covariance
experiment_name = "experiments-1b-8-gaussianLocation-missspec-withV"
output_path_1b_8 = output_path_1b*experiment_name*"/"
isdir(output_path_1b_8) || mkdir(output_path_1b_8)

parmdim = 10
Σ = Diagonal(0.5*ones(parmdim)).+0.5
Σlik = Matrix(Diagonal(sqrt.(1:parmdim)))



samplesize = 1000

iteravg_numepochs=8 # NEW
num_iteravgs = 10000
numepochs = num_iteravgs*iteravg_numepochs

w=1/2
pows = [1,2/3,1/2,1/3] # Not actually used, just a reference for JN & HF
algos = [:vanillaSGDPowOneSumOne, :vanillaSGDPowTwoThirdSumOne, :vanillaSGDPowHalfSumOne, :vanillaSGDPowThirdSumOne]
batchsizes = Dict(
    :vanillaSGDPowOneSumOne => 1*ceil(Int,samplesize^(1-1)),
    :vanillaSGDPowTwoThirdSumOne => 1*ceil(Int,samplesize^(1-2/3)),
    :vanillaSGDPowHalfSumOne => 1*ceil(Int,samplesize^(1-1/2)),
    :vanillaSGDPowThirdSumOne => 1*ceil(Int,samplesize^(1-1/3))
)
numstepss = Dict(
    algo => numepochs*samplesize ÷ batchsizes[algo] for algo in algos
)

layers = layerer([:sandwichScaled, :vanillaSGDPowOneSumOne, :iteravg, :trueparm])
layers = layerer(layers, :vanillaSGDPowTwoThirdSumOne, drop=:vanillaSGDPowOneSumOne)
layers = layerer(layers, :vanillaSGDPowHalfSumOne, drop=:vanillaSGDPowTwoThirdSumOne)
layers = layerer(layers, :vanillaSGDPowThirdSumOne, drop=:vanillaSGDPowHalfSumOne)


gaussianLocationInit(samplesize, parmdim, Σ, Σlik, output_path_1b_8*"model_init.jld2")
modelRun(output_path_1b_8*"model_init.jld2", output_path_1b_8*"model_run.jld2",
    algos, samplesize, batchsizes, w, numstepss; multithread=true)
modelPlot(output_path_1b_8*"model_run.jld2", output_path_1b_8*experiment_name*"-fig",
    layers, (-3,3), (-3,3), nameattr= false, iteravg_numepochs=iteravg_numepochs)
