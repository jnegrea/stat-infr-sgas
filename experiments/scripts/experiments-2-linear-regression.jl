using LinearAlgebra

source_path = "../core/"
include(source_path*"sgld-module-0.jl")
# using .SGLDModule

output_path = "../output/"
isdir(output_path) || mkdir(output_path)
## experiment-1 Gausian Location Experiments

output_path_2 = output_path*"experiments-2-linearRegression/"
isdir(output_path_2) || mkdir(output_path_2)

## Example 2.1: Isotropic Covariates
experiment_name = "experiments-2-1-spherical-covariance"
output_path_2_1 = output_path_2*experiment_name*"/"
isdir(output_path_2_1) || mkdir(output_path_2_1)

samplesize = 100
parmdim = 10
w=1/2
Σcov = Matrix(Diagonal(ones(parmdim)))
σ2 = 1
β = (1:parmdim) .- parmdim ÷ 2

algos = [:vanillaSGD, :preconSGD, :preconSGLD, :largeBatchPreconSGLD]

batchsizes = Dict(
    :vanillaSGD => 1,
    :preconSGD => 1,
    :preconSGLD => 1,
    :largeBatchPreconSGLD => samplesize÷4
)

numstepsfactor = 10000
numstepss = Dict(
    :vanillaSGD => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGD], 
    :preconSGD => numstepsfactor*samplesize ÷ batchsizes[:preconSGD],
    :preconSGLD => numstepsfactor*samplesize ÷ batchsizes[:preconSGLD],
    :largeBatchPreconSGLD => 20*numstepsfactor*samplesize ÷ batchsizes[:largeBatchPreconSGLD]
)

layers = layerer([:halfsandwich, :sandwich])
layers = layerer(layers, :vanillaSGD)
layers = layerer(layers, :trueparm)
layers = layerer(layers, :preconSGD; drop=:vanillaSGD)
layers = layerer(layers, [:preconSGLD, :mixture, :jinv]; drops=[:preconSGD, :halfsandwich])
layers = layerer(layers, [:largeBatchPreconSGLD]; drops=[:preconSGLD,:mixture])

# linearRegressionInit(samplesize, parmdim, Σcov, β,σ2, output_path_2_1*"model_init.jld2")
# modelRun(output_path_2_1*"model_init.jld2", output_path_2_1*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true)
# modelPlot(output_path_2_1*"model_run.jld2", output_path_2_1*experiment_name*"-fig", 
#     layers, (-3,3), (-3,3), nameattr= false)
# modelSummary(output_path_2_1*"model_init.jld2", output_path_2_1*"model_run.jld2", 
#     output_path_2_1*experiment_name*"-summary.txt")

## Example 2.2: Anisotropic Cvoariates
experiment_name = "experiments-2-2-anisotropic-covariance"
output_path_2_2 = output_path_2*experiment_name*"/"
isdir(output_path_2_2) || mkdir(output_path_2_2)

samplesize = 1000
parmdim = 10
w=1/2
s=0.1
Σcov = 1/4*inv(s*Matrix(Diagonal(ones(parmdim))).+(1-s))
Σcov = sqrt(Σcov*Σcov')
σ2 = 1/4
β = ones(parmdim)

algos = [:vanillaSGD, :preconSGD, :preconSGLD]

batchsizes = Dict(
    :vanillaSGD => 1,
    :preconSGD => 1,
    :preconSGLD => 1
    # :largeBatchPreconSGLD => samplesize÷4
)

numstepsfactor = 10000
numstepss = Dict(
    :vanillaSGD => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGD], 
    :preconSGD => numstepsfactor*samplesize ÷ batchsizes[:preconSGD],
    :preconSGLD => numstepsfactor*samplesize ÷ batchsizes[:preconSGLD]
    # :largeBatchPreconSGLD => 20*numstepsfactor*samplesize ÷ batchsizes[:largeBatchPreconSGLD]
)

layers = layerer([:halfsandwich, :sandwich])
layers = layerer(layers, :vanillaSGD)
layers = layerer(layers, :trueparm)
layers = layerer(layers, :preconSGD; drop=:vanillaSGD)
layers = layerer(layers, [:preconSGLD, :mixture]; drops=[:preconSGD, :halfsandwich])
# layers = layerer(layers, [:largeBatchPreconSGLD]; drops=[:preconSGLD,:mixture])

# linearRegressionInit(samplesize, parmdim, Σcov, β,σ2, output_path_2_2*"model_init.jld2")
# modelRun(output_path_2_2*"model_init.jld2", output_path_2_2*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true)
# modelPlot(output_path_2_2*"model_run.jld2", output_path_2_2*experiment_name*"-fig", 
#     layers, (-3,3), (-3,3), nameattr= false)
# modelSummary(output_path_2_2*"model_init.jld2", output_path_2_2*"model_run.jld2", 
#     output_path_2_2*experiment_name*"-summary.txt")

## Example 2.3: Anisotropic Cvoariates -- test univariate plots
experiment_name = "experiments-2-3-anisotropic-covariance-test-univ-plots"
output_path_2_3 = output_path_2*experiment_name*"/"
isdir(output_path_2_3) || mkdir(output_path_2_3)

samplesize = 100
parmdim = 10
w=1/2
s=0.1
Σcov = 1/4*inv(s*Matrix(Diagonal(ones(parmdim))).+(1-s))
Σcov = sqrt(Σcov*Σcov')
σ2 = 1/4
β = ones(parmdim)

algos = [:vanillaSGD, :preconSGD, :preconSGLD, :vanillaSGLDdecreasing]

batchsizes = Dict(
    :vanillaSGD => 1,
    :preconSGD => 1,
    :preconSGLD => 1,
    :vanillaSGLDdecreasing => 1
    # :largeBatchPreconSGLD => samplesize÷4
)

numstepsfactor = 1000
numstepss = Dict(
    :vanillaSGD => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGD], 
    :preconSGD => numstepsfactor*samplesize ÷ batchsizes[:preconSGD],
    :preconSGLD => numstepsfactor*samplesize ÷ batchsizes[:preconSGLD],
    :vanillaSGLDdecreasing => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGLDdecreasing]
    # :largeBatchPreconSGLD => 20*numstepsfactor*samplesize ÷ batchsizes[:largeBatchPreconSGLD]
)

layers = layerer([:halfsandwich, :sandwich, :univariate])
layers = layerer(layers, :vanillaSGD)
layers = layerer(layers, :trueparm)
layers = layerer(layers, :preconSGD; drop=:vanillaSGD)
layers = layerer(layers, [:preconSGLD, :mixture]; drops=[:preconSGD, :halfsandwich])
layers = layerer(layers, [:vanillaSGLDdecreasing, :jinv]; drops=[:preconSGLD, :mixture])
# layers = layerer(layers, [:largeBatchPreconSGLD]; drops=[:preconSGLD,:mixture])

linearRegressionInit(samplesize, parmdim, Σcov, β,σ2, output_path_2_3*"model_init.jld2")
modelRun(output_path_2_3*"model_init.jld2", output_path_2_3*"model_run.jld2",
    algos, samplesize, batchsizes, w, numstepss; multithread=false)
modelPlot(output_path_2_3*"model_run.jld2", output_path_2_3*experiment_name*"-fig", 
    layers, (-3,3), (-3,3), nameattr= false)
modelSummary(output_path_2_3*"model_init.jld2", output_path_2_3*"model_run.jld2", 
    output_path_2_3*experiment_name*"-summary.txt")
