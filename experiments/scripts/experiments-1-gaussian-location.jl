using LinearAlgebra

source_path = "../core/"
include(source_path*"sgld-module-0.jl")
# using .SGLDModule

output_path = "../output/"
isdir(output_path) || mkdir(output_path)
## experiment-1 Gausian Location Experiments

output_path_1 = output_path*"experiments-1-gaussianLocation/"
isdir(output_path_1) || mkdir(output_path_1)

## Example 1.1: Well-spec gaussian location -- a simple test case to make sure everything works
experiment_name = "experiments-1-1-gaussianLocation-wellspec"
output_path_1_1 = output_path_1*experiment_name*"/"
isdir(output_path_1_1) || mkdir(output_path_1_1)

samplesize = 100
batchsize = 1
parmdim = 10
w=1/2
Σ = Matrix(Diagonal(ones(parmdim)))
Σlik = Matrix(Diagonal(ones(parmdim)))

algos = [:vanillaSGD, :preconSGD]
batchsizes = Dict(
    :vanillaSGD => 1,
    :preconSGD => 1
)
numstepss = Dict(
    :vanillaSGD => 10000*samplesize ÷ batchsizes[:vanillaSGD], 
    :preconSGD => 10000*samplesize ÷ batchsizes[:preconSGD]
)

layers = layerer([:halfsandwich, :sandwich])
layers = layerer(layers, :vanillaSGD)
layers = layerer(layers, :trueparm)


# gaussianLocationInit(samplesize, parmdim, Σ, Σlik, output_path_1_1*"model_init.jld2")
# modelRun(output_path_1_1*"model_init.jld2", output_path_1_1*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true)
# modelPlot(output_path_1_1*"model_run.jld2", output_path_1_1*experiment_name*"-fig", 
#     layers, (-3,3), (-3,3), nameattr= false)
# modelSummary(output_path_1_1*"model_init.jld2", output_path_1_1*"model_run.jld2", 
#     output_path_1_1*experiment_name*"-summary.txt")

## Example 1.2: well-spec multivariate gaussian location -- non-spherical
experiment_name = "experiments-1-2-gaussianLocation-wellspec-nonspherical"
output_path_1_2 = output_path_1*experiment_name*"/"
isdir(output_path_1_2) || mkdir(output_path_1_2)

samplesize = 100
batchsize = 1
parmdim = 10
w=1/2
Σ = Matrix(Diagonal(1.0 * (1.0:(parmdim))))
Σlik = Matrix(Diagonal(1.0 * (1.0:(parmdim))))

algos = [:vanillaSGD, :preconSGD]
batchsizes = Dict(
    :vanillaSGD => 1,
    :preconSGD => 1
)
numstepss = Dict(
    :vanillaSGD => 10000*samplesize ÷ batchsizes[:vanillaSGD], 
    :preconSGD => 10000*samplesize ÷ batchsizes[:preconSGD]
)

layers = layerer([:halfsandwich])
layers = layerer(layers, :vanillaSGD)
layers = layerer(layers, :sandwich)
layers = layerer(layers, :preconSGD)
layers = layerer(layers, :trueparm)


# gaussianLocationInit(samplesize, parmdim, Σ, Σlik, output_path_1_2*"model_init.jld2")
# modelRun(output_path_1_2*"model_init.jld2", output_path_1_2*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true)
# modelPlot(output_path_1_2*"model_run.jld2", output_path_1_2*experiment_name*"-fig", 
#     layers, (-3,3), (-3,3), nameattr= false)
# modelSummary(output_path_1_2*"model_init.jld2", output_path_1_2*"model_run.jld2", 
#     output_path_1_2*experiment_name*"-summary.txt")

## Example 1.3: miss-spec multivariate gaussian location with wrong covariance 
experiment_name = "experiments-1-3-gaussianLocation-missspec"
output_path_1_3 = output_path_1*experiment_name*"/"
isdir(output_path_1_3) || mkdir(output_path_1_3)

samplesize = 100
batchsize = 2
parmdim = 10
w=1/2
Σ = Diagonal(0.5*ones(parmdim)).+0.5
Σlik = Matrix(Diagonal(1:parmdim))

algos = [:vanillaSGD, :preconSGD]
batchsizes = Dict(
    :vanillaSGD => 1,
    :preconSGD => 1
)
numstepss = Dict(
    :vanillaSGD => 10000*samplesize ÷ batchsizes[:vanillaSGD], 
    :preconSGD => 10000*samplesize ÷ batchsizes[:preconSGD]
)

layers = layerer([:halfsandwich, :vanillaSGD])
layers = layerer(layers, [:sandwich,:preconSGD])
layers = layerer(layers, :trueparm)


# gaussianLocationInit(samplesize, parmdim, Σ, Σlik, output_path_1_3*"model_init.jld2")
# modelRun(output_path_1_3*"model_init.jld2", output_path_1_3*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true)
# modelPlot(output_path_1_3*"model_run.jld2", output_path_1_3*experiment_name*"-fig", 
#     layers, (-3,3), (-3,3), nameattr= false)
# modelSummary(output_path_1_3*"model_init.jld2", output_path_1_3*"model_run.jld2", 
#     output_path_1_3*experiment_name*"-summary.txt")

## Example 1.4: miss-spec multivariate gaussian location with wrong covariance, non-diagonal
experiment_name = "experiments-1-4-gaussianLocation-missspec-non-diagonal"
output_path_1_4 = output_path_1*experiment_name*"/"
isdir(output_path_1_4) || mkdir(output_path_1_4)

samplesize = 100
batchsize = 1
parmdim = 10
w=1/2
Σ = Matrix(Diagonal(ones(parmdim))) 
Σlik = Matrix(Diagonal(1.0 * (1.0:(parmdim))))

algos = [:vanillaSGD, :preconSGD, :preconSGLD, :largeBatchPreconSGLD]
batchsizes = Dict(
    :vanillaSGD => 1,
    :preconSGD => 1,
    :preconSGLD => 1,
    :largeBatchPreconSGLD => samplesize÷4
)
numstepss = Dict(
    :vanillaSGD => 10000*samplesize ÷ batchsizes[:vanillaSGD], 
    :preconSGD => 10000*samplesize ÷ batchsizes[:preconSGD],
    :preconSGLD => 10000*samplesize ÷ batchsizes[:preconSGLD],
    :largeBatchPreconSGLD => 200000*samplesize ÷ batchsizes[:largeBatchPreconSGLD]
)

layers = layerer([:halfsandwich, :sandwich, :vanillaSGD])
layers = layerer(layers, :trueparm)
layers = layerer(layers, :preconSGD; drop=:vanillaSGD)


# gaussianLocationInit(samplesize, parmdim, Σ, Σlik, output_path_1_4*"model_init.jld2")
# modelRun(output_path_1_4*"model_init.jld2", output_path_1_4*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true)
# modelPlot(output_path_1_4*"model_run.jld2", output_path_1_4*experiment_name*"-fig", 
#     layers, (-3,3), (-3,3), nameattr= false)
# modelSummary(output_path_1_4*"model_init.jld2", output_path_1_4*"model_run.jld2", 
#     output_path_1_4*experiment_name*"-summary.txt")
    
layers = layerer([:sandwich, :jinv, :mixture])
layers = layerer(layers, :preconSGLD)
layers = layerer(layers, :trueparm)

# modelPlot(output_path_1_4*"model_run.jld2", output_path_1_4*experiment_name*"-fig-mixture-", 
#     layers, (-3,3), (-3,3), nameattr= false)
# 
# layers = layerer([:jinv, :sandwich, :largeBatchPreconSGLD])
# layers = layerer(layers, :trueparm)
# 
# modelPlot(output_path_1_4*"model_run.jld2", output_path_1_4*experiment_name*"-fig-largeBatch-", 
#     layers, (-3,3), (-3,3), nameattr= false)

## Example 1.5: well-spec multivariate gaussian location with non-diagonal covariance
experiment_name = "experiments-1-5-gaussianLocation-missspec"
output_path_1_5 = output_path_1*experiment_name*"/"
isdir(output_path_1_5) || mkdir(output_path_1_5)

samplesize = 100
batchsize = 2
parmdim = 10
w=1/2
Σ = 5*(Diagonal(0.1*ones(parmdim)).+0.9)
Σlik = Σ

algos = [:vanillaSGD, :preconSGD]
batchsizes = Dict(
    :vanillaSGD => 1,
    :preconSGD => 1
)
numstepss = Dict(
    :vanillaSGD => 10000*samplesize ÷ batchsizes[:vanillaSGD], 
    :preconSGD => 10000*samplesize ÷ batchsizes[:preconSGD]
)

layers = layerer([:halfsandwich, :jinv])
layers = layerer(layers, :vanillaSGD)
layers = layerer(layers, :trueparm)
layers = layerer(layers, :preconSGD; drop=:vanillaSGD)

# gaussianLocationInit(samplesize, parmdim, Σ, Σlik, output_path_1_5*"model_init.jld2")
# modelRun(output_path_1_5*"model_init.jld2", output_path_1_5*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true)
# modelPlot(output_path_1_5*"model_run.jld2", output_path_1_5*experiment_name*"-fig", 
#     layers, (-3,3), (-3,3), nameattr= false)
# modelSummary(output_path_1_5*"model_init.jld2", output_path_1_5*"model_run.jld2", 
#     output_path_1_5*experiment_name*"-summary.txt")

## Example 1.6: well-spec multivariate gaussian location with non-diagonal covariance for presentation
experiment_name = "experiments-1-6-gaussianLocation-missspec-presentation"
output_path_1_6 = output_path_1*experiment_name*"/"
isdir(output_path_1_6) || mkdir(output_path_1_6)

samplesize = 100
parmdim = 10
w=1/2
# s=0.25
# Σ = 0.5*(Diagonal(s*ones(parmdim)).+(1-s))
# Σlik = copy(Σ)
# Σlik[2:parmdim,:] = -Σlik[2:parmdim,:]
# Σlik[:,2:parmdim] = -Σlik[:,2:parmdim]

s=0.1
Σlik = 0.125*(Diagonal(ones(parmdim))) 
Σ = (Diagonal(s*ones(parmdim)).+(1-s))


# algos = [:vanillaSGLD, :preconSGD, :preconSGLD, :vanillaSGLDdecreasing, :largeBatchPreconSGLD]
algos = [:vanillaSGLD, :preconSGD, :preconSGLD]

batchsizes = Dict(
    :vanillaSGLD => 1,
    :preconSGD => 1,
    :preconSGLD => 1
    # :vanillaSGLDdecreasing => 1,
    # :largeBatchPreconSGLD => samplesize÷4
)

numstepsfactor = 10000
numstepss = Dict(
    :vanillaSGLD => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGLD], 
    :preconSGD => numstepsfactor*samplesize ÷ batchsizes[:preconSGD],
    :preconSGLD => numstepsfactor*samplesize ÷ batchsizes[:preconSGLD]
    # :vanillaSGLDdecreasing => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGLDdecreasing]
    # :largeBatchPreconSGLD => numstepsfactor*samplesize ÷ batchsizes[:largeBatchPreconSGLD]
)

layers = layerer([:sandwich, :jinv])
layers = layerer(layers, :trueparm)
layers = layerer(layers, :vanillaSGLD)
layers = layerer(layers, [:preconSGLD, :mixture]; drop=:vanillaSGLD)
# layers = layerer(layers, [:vanillaSGLDdecreasing]; drops=[:preconSGLD, :mixture])
# layers = layerer(layers, [:largeBatchPreconSGLD]; drops=[:vanillaSGLDdecreasing])

# gaussianLocationInit(samplesize, parmdim, Σ, Σlik, output_path_1_6*"model_init.jld2")
# modelRun(output_path_1_6*"model_init.jld2", output_path_1_6*"model_run",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true)
# modelPlot(output_path_1_6*"model_run.jld2", output_path_1_6*experiment_name*"-fig", 
    # layers, (-3,3), (-3,3), nameattr= false, levels = 6)
# modelSummary(output_path_1_6*"model_init.jld2", output_path_1_6*"model_run.jld2", 
#     output_path_1_6*experiment_name*"-summary.txt")

layers = layerer([:sandwich, :jinv, :nolegend, :noaxes])
layers = layerer(layers, :trueparm)
layers = layerer(layers, :vanillaSGLD)
layers = layerer(layers, [:preconSGLD, :mixture]; drop=:vanillaSGLD)
# modelPlot(output_path_1_6*"model_run.jld2", output_path_1_6*experiment_name*"nolegend-noaxes-fig", 
    # layers, (-3,3), (-3,3), nameattr= false, levels = 6)

layers = layerer([:sandwich, :jinv, :univariate, :histogram])
layers = layerer(layers, :trueparm)
layers = layerer(layers, :vanillaSGLD)
layers = layerer(layers, [:preconSGLD, :mixture])
# modelPlot(output_path_1_6*"model_run.jld2", output_path_1_6*experiment_name*"-univariate-fig", 
    # layers, (-3,3), (-3,3), nameattr= false, levels = 6, localscale=false)
        

output_path_1_6_1= output_path_1_6*"/all-parms-univariate/"
isdir(output_path_1_6_1) || mkdir(output_path_1_6_1)

layers = layerer([:sandwich, :jinv, :allparms, :univariate, :histogram])
layers = layerer(layers, :trueparm)
layers = layerer(layers, :vanillaSGLD)
layers = layerer(layers, [:preconSGLD, :mixture])
# modelPlot(output_path_1_6*"model_run.jld2", output_path_1_6_1*experiment_name*"-fig", 
#     layers, (-3,3), (-3,3), nameattr= false, levels = 6, localscale=false)
    
    
## Example 1.7: well-spec multivariate gaussian location with non-diagonal covariance for presentation
experiment_name = "experiments-1-7-gaussianLocation-missspec-presentation-jsm"
output_path_1_7 = output_path_1*experiment_name*"/"
isdir(output_path_1_7) || mkdir(output_path_1_7)

samplesize = 100
parmdim = 10
w=1/2
# s=0.25
# Σ = 0.5*(Diagonal(s*ones(parmdim)).+(1-s))
# Σlik = copy(Σ)
# Σlik[2:parmdim,:] = -Σlik[2:parmdim,:]
# Σlik[:,2:parmdim] = -Σlik[:,2:parmdim]

s=0.1
Σlik = 0.125*(Diagonal(ones(parmdim))) 
Σ = (Diagonal(s*ones(parmdim)).+(1-s))


# algos = [:vanillaSGLD, :preconSGD, :preconSGLD, :vanillaSGLDdecreasing, :largeBatchPreconSGLD]
algos = [:vanillaSGD, :preconSGD, :vanillaSGDPow, :preconSGDPow]

batchsizes = Dict(
    :vanillaSGD => 1,
    :preconSGD => 1,
    :vanillaSGDPow => 1,
    :preconSGDPow => 1
    # :vanillaSGLDdecreasing => 1,
    # :largeBatchPreconSGLD => samplesize÷4
)

numstepsfactor = 10000
numstepss = Dict(
    :vanillaSGD => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGD], 
    :preconSGD => numstepsfactor*samplesize ÷ batchsizes[:preconSGD],
    :vanillaSGDPow => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGDPow],
    :preconSGDPow => numstepsfactor*samplesize ÷ batchsizes[:preconSGDPow]
    # :vanillaSGLDdecreasing => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGLDdecreasing]
    # :largeBatchPreconSGLD => numstepsfactor*samplesize ÷ batchsizes[:largeBatchPreconSGLD]
)



# gaussianLocationInit(samplesize, parmdim, Σ, Σlik, output_path_1_7*"model_init.jld2")
# modelRun(output_path_1_7*"model_init.jld2", output_path_1_7*"model_run",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true)
# modelSummary(output_path_1_7*"model_init.jld2", output_path_1_7*"model_run.jld2", 
#     output_path_1_7*experiment_name*"-summary.txt")
    
layers = layerer([:sandwich, :jinv])
layers = layerer(layers, :trueparm)
layers = layerer(layers, :vanillaSGD)
layers = layerer(layers, [:iteravg,:vanillaSGDPow])
layers = layerer(layers, [:preconSGD]; drop=:vanillaSGD)
# layers = layerer(layers, [:vanillaSGLDdecreasing]; drops=[:preconSGLD, :mixture])
# layers = layerer(layers, [:largeBatchPreconSGLD]; drops=[:vanillaSGLDdecreasing])    
# modelPlot(output_path_1_7*"model_run.jld2", output_path_1_7*experiment_name*"-fig", 
#     layers, (-3,3), (-3,3), nameattr= false, levels = 6)


layers = layerer([:sandwich, :jinv, :nolegend, :noaxes])
layers = layerer(layers, :trueparm)
layers = layerer(layers, :vanillaSGD)
layers = layerer(layers, [:iteravg,:vanillaSGDPow])
layers = layerer(layers, [:preconSGD]; drop=:vanillaSGD)
# modelPlot(output_path_1_7*"model_run.jld2", output_path_1_7*experiment_name*"nolegend-noaxes-fig", 
#     layers, (-3,3), (-3,3), nameattr= false, levels = 6)

layers = layerer([:sandwich, :jinv, :univariate, :histogram])
layers = layerer(layers, :trueparm)
layers = layerer(layers, :vanillaSGD)
layers = layerer(layers, [:iteravg,:vanillaSGDPow])
layers = layerer(layers, [:preconSGD])
# modelPlot(output_path_1_7*"model_run.jld2", output_path_1_7*experiment_name*"-univariate-fig", 
    # layers, (-3,3), (-3,3), nameattr= false, levels = 6, localscale=false)
        

output_path_1_7_1= output_path_1_7*"/all-parms-univariate/"
isdir(output_path_1_7_1) || mkdir(output_path_1_7_1)

layers = layerer([:sandwich, :jinv, :allparms, :univariate, :histogram])
layers = layerer(layers, :trueparm)
layers = layerer(layers, :vanillaSGD)
layers = layerer(layers, [:iteravg,:vanillaSGDPow])
layers = layerer(layers, [:preconSGD])
# modelPlot(output_path_1_7*"model_run.jld2", output_path_1_7_1*experiment_name*"-fig", 
    # layers, (-3,3), (-3,3), nameattr= false, levels = 6, localscale=false)
    
## Example 1.8: miss-spec multivariate gaussian location with wrong covariance 
experiment_name = "experiments-1-8-gaussianLocation-missspec-withV"
output_path_1_8 = output_path_1*experiment_name*"/"
isdir(output_path_1_8) || mkdir(output_path_1_8)

samplesize = 1000
parmdim = 10
w=1/2
Σ = Diagonal(0.5*ones(parmdim)).+0.5
Σlik = Matrix(Diagonal(sqrt.(1:parmdim)))

algos = [:vanillaSGD, :preconSGD, :preconVSGD, :preconSGLD]
batchsizes = Dict(
    :vanillaSGD => 1,
    :preconSGD => 1,
    :preconVSGD => 1,
    :preconSGLD => 1
)
numstepss = Dict(
    :vanillaSGD => 10000*samplesize ÷ batchsizes[:vanillaSGD], 
    :preconSGD => 10000*samplesize ÷ batchsizes[:preconSGD],
    :preconVSGD => 10000*samplesize ÷ batchsizes[:preconVSGD],
    :preconSGLD => 10000*samplesize ÷ batchsizes[:preconSGLD],
)

layers = layerer([:halfsandwich, :vanillaSGD, :trueparm, :nogrid])
layers = layerer(layers, [:sandwich,:preconSGD], drop=[:halfsandwich, :vanillaSGD])
layers = layerer(layers, [:jinv,:preconVSGD], drop=[:preconSGD,:sandwich])
layers = layerer(layers, [:mixture,:preconSGLD], drop=[:jinv,:preconVSGD])



# gaussianLocationInit(samplesize, parmdim, Σ, Σlik, output_path_1_8*"model_init.jld2")
# modelRun(output_path_1_8*"model_init.jld2", output_path_1_8*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true)
modelPlot(output_path_1_8*"model_run.jld2", output_path_1_8*experiment_name*"-fig", 
    layers, (-3,3), (-3,3), nameattr= false, levels = 5, fontscale=2.2, legendloc=:outertop)
# modelSummary(output_path_1_8*"model_init.jld2", output_path_1_8*"model_run.jld2", 
    # output_path_1_8*experiment_name*"-summary.txt")