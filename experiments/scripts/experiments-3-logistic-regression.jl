using LinearAlgebra

source_path = "../core/"
include(source_path*"sgld-module-0.jl")
# using .SGLDModule

output_path = "../output/"
isdir(output_path) || mkdir(output_path)
## experiment-3 Logistic REgression Experiments

output_path_3 = output_path*"experiments-3-logisticRegression/"
isdir(output_path_3) || mkdir(output_path_3)

## Example 3.1: Isotropic Covariates
experiment_name = "experiments-3-1-spherical-covariance"
output_path_3_1 = output_path_3*experiment_name*"/"
isdir(output_path_3_1) || mkdir(output_path_3_1)

samplesize = 10000
batchsize = 100
parmdim = 10
w=1/2
Σcov = Matrix(Diagonal(ones(parmdim)))
σ2 = 1
β = ((1:parmdim) .- parmdim ÷ 2)./2 ./parmdim

algos = [:vanillaSGD, :preconSGD, :preconSGLD]

batchsizes = Dict(
    :vanillaSGD => 100,
    :preconSGD => 100,
    :preconSGLD => 100
    # :largeBatchPreconSGLD => samplesize÷4
)

numstepsfactor = 1000
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
layers = layerer(layers, [:preconSGLD, :mixture, :jinv]; drops=[:preconSGD, :halfsandwich])
# layers = layerer(layers, [:largeBatchPreconSGLD]; drops=[:preconSGLD,:mixture])

# logisticRegressionInit(samplesize, parmdim, Σcov, β, output_path_3_1*"model_init.jld2"; outof=:one, mc_size=Int(1e6))
# modelRun(output_path_3_1*"model_init.jld2", output_path_3_1*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true)
# modelPlot(output_path_3_1*"model_run.jld2", output_path_3_1*experiment_name*"-fig", 
#     layers, (-3,3), (-3,3), nameattr= false, levels=10)
# modelSummary(output_path_3_1*"model_init.jld2", output_path_3_1*"model_run.jld2", 
    # output_path_3_1*experiment_name*"-summary.txt")

## Example 3.2: Anisotropic Covariates
experiment_name = "experiments-3-2-anisotropic-covariance"
output_path_3_2 = output_path_3*experiment_name*"/"
isdir(output_path_3_2) || mkdir(output_path_3_2)

samplesize = 1000
batchsize = 1
parmdim = 10
w=1/2
Σcov = Matrix(Diagonal((1:parmdim)./sqrt(parmdim)))
σ2 = 1
β = ((1:parmdim) .- parmdim ÷ 2)./2 ./parmdim

algos = [:vanillaSGD, :preconSGD, :preconSGLD]

batchsizes = Dict(
    :vanillaSGD => 1,
    :preconSGD => 1,
    :preconSGLD => 1
    # :largeBatchPreconSGLD => samplesize÷4
)

numstepsfactor = 1000
numstepss = Dict(
    :vanillaSGD => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGD], 
    :preconSGD => numstepsfactor*samplesize ÷ batchsizes[:preconSGD],
    :preconSGLD => numstepsfactor*samplesize ÷ batchsizes[:preconSGLD]
    # :largeBatchPreconSGLD => numstepsfactor*samplesize ÷ batchsizes[:largeBatchPreconSGLD]
)

layers = layerer([:halfsandwich, :sandwich])
layers = layerer(layers, :vanillaSGD)
layers = layerer(layers, :trueparm)
layers = layerer(layers, :preconSGD; drop=:vanillaSGD)
layers = layerer(layers, [:preconSGLD, :mixture, :jinv]; drops=[:preconSGD, :halfsandwich])
# layers = layerer(layers, [:largeBatchPreconSGLD]; drops=[:preconSGLD,:mixture])

logisticRegressionInit(samplesize, parmdim, Σcov, β, output_path_3_2*"model_init.jld2"; outof=(:poisson,10))
modelRun(output_path_3_2*"model_init.jld2", output_path_3_2*"model_run.jld2",
    algos, samplesize, batchsizes, w, numstepss; multithread=true)
modelPlot(output_path_3_2*"model_run.jld2", output_path_3_2*experiment_name*"-fig", 
    layers, (-3,3), (-3,3), nameattr= false, levels=10)
# modelSummary(output_path_3_2*"model_init.jld2", output_path_3_2*"model_run.jld2", 
#     output_path_3_2*experiment_name*"-summary.txt")


## Example 3.3: Isotropic Covariates HugeSample
experiment_name = "experiments-3-3-spherical-covariance-huge"
output_path_3_3 = output_path_3*experiment_name*"/"
isdir(output_path_3_3) || mkdir(output_path_3_3)

# samplesize = 120000000
samplesize = 100000

batchsize = 100
parmdim = 10
w=1/2
Σcov = Matrix(Diagonal(ones(parmdim)))
σ2 = 1
β = ((1:parmdim) .- parmdim ÷ 2)./2 ./parmdim

algos = [:vanillaSGD, :preconSGD, :preconSGLD]

batchsizes = Dict(
    :vanillaSGD => 100,
    :preconSGD => 100,
    :preconSGLD => 100
)

numstepsfactor = 10000
numstepss = Dict(
    :vanillaSGD => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGD], 
    :preconSGD => numstepsfactor*samplesize ÷ batchsizes[:preconSGD],
    :preconSGLD => numstepsfactor*samplesize ÷ batchsizes[:preconSGLD]
)

layers = layerer([:halfsandwich, :sandwich])
layers = layerer(layers, :vanillaSGD)
layers = layerer(layers, :trueparm)
layers = layerer(layers, :preconSGD; drop=:vanillaSGD)
layers = layerer(layers, [:preconSGLD, :mixture, :jinv]; drops=[:preconSGD, :halfsandwich])

# logisticRegressionInitHuge(samplesize, parmdim, Σcov, β, output_path_3_3*"model_init"; outof=:one, mc_size=Int(1e6), atatime=1000)
# modelRun(output_path_3_3*"model_init.jld2", output_path_3_3*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=false)
# modelPlot(output_path_3_3*"model_run.jld2", output_path_3_3*experiment_name*"-fig", 
#     layers, (-6,6), (-6,6), nameattr= false, levels = 15)
# modelSummary(output_path_3_3*"model_init.jld2", output_path_3_3*"model_run.jld2", 
#     output_path_3_3*experiment_name*"-summary.txt")


## Example 3.4: Anisotropic Covariates
experiment_name = "experiments-3-4-anisotropic-covariance-withV"
output_path_3_4 = output_path_3*experiment_name*"/"
isdir(output_path_3_4) || mkdir(output_path_3_4)

# samplesize = 10
samplesize = 100000
parmdim = 10
w=1/2
Σcov = Matrix(Diagonal((1:parmdim)./sqrt(parmdim)))
σ2 = 1
β = ((1:parmdim) .- parmdim ÷ 2)./2 ./parmdim

algos = [:vanillaSGD, :preconSGD, :preconVSGD, :preconSGLD]

batchsizes = Dict(
    :vanillaSGD => 200,
    :preconSGD => 200,
    :preconVSGD => 200,
    :preconSGLD => 200)

numstepsfactor = 10000
numstepss = Dict(
    :vanillaSGD => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGD], 
    :preconSGD => numstepsfactor*samplesize ÷ batchsizes[:preconSGD],
    :preconVSGD => numstepsfactor*samplesize ÷ batchsizes[:preconVSGD],
    :preconSGLD => numstepsfactor*samplesize ÷ batchsizes[:preconSGLD]
)

layers = layerer([:halfsandwich, :vanillaSGD, :trueparm])
layers = layerer(layers, [:sandwich,:preconSGD], drop=[:halfsandwich, :vanillaSGD])
layers = layerer(layers, [:jinv,:preconVSGD], drop=[:preconSGD,:sandwich])
layers = layerer(layers, [:mixture,:preconSGLD], drop=[:jinv,:preconVSGD])

# logisticRegressionInit(samplesize, parmdim, Σcov, β, output_path_3_4*"model_init.jld2"; outof=(:poisson,10), mc_size=Int(1e5))
# modelRun(output_path_3_4*"model_init.jld2", output_path_3_4*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true)
# modelPlot(output_path_3_4*"model_run.jld2", output_path_3_4*experiment_name*"-fig", 
#     layers, (-3,3), (-3,3), nameattr= false, levels = 5)
# modelSummary(output_path_3_4*"model_init.jld2", output_path_3_4*"model_run.jld2", 
#     output_path_3_4*experiment_name*"-summary.txt")


## Example 3.5: Iteravg test
experiment_name = "experiments-3-5-logistic-iteravg-test"
output_path_3_5 = output_path_3*experiment_name*"/"
isdir(output_path_3_5) || mkdir(output_path_3_5)


samplesize = 100000
parmdim = 10
w=1/2
Σcov = Matrix(Diagonal((1:parmdim)./sqrt(parmdim)))
σ2 = 1
β = ((1:parmdim) .- parmdim ÷ 2)./2 ./parmdim


algos = [:vanillaSGDPowHalf, :vanillaSGDPowTwoThird, :vanillaSGDPowThird]

batchsizes = Dict(
    :vanillaSGDPowHalf => 100,
    :vanillaSGDPowTwoThird => 100,
    :vanillaSGDPowThird => 100)

numstepsfactor = 10000
numstepss = Dict(
    :vanillaSGDPowHalf => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGDPowHalf], 
    :vanillaSGDPowTwoThird => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGDPowTwoThird], 
    :vanillaSGDPowThird => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGDPowThird])
    
keepperiods = Dict(
    :vanillaSGDPowHalf => 1, 
    :vanillaSGDPowTwoThird => 1,
    :vanillaSGDPowThird => 1)

# logisticRegressionInit(samplesize, parmdim, Σcov, β, output_path_3_5*"model_init.jld2"; outof=(:poisson,10), mc_size=Int(1e5))
# modelRun(output_path_3_5*"model_init.jld2", output_path_3_5*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true)



layers = layerer(:sandwich)
layers = layerer(layers, [:vanillaSGDPowHalf, :iteravg])
layers = layerer(layers, :trueparm)
layers = layerer(layers, :vanillaSGDPowTwoThird; drop=:vanillaSGDPowHalf)
layers = layerer(layers, :vanillaSGDPowThird; drop=:vanillaSGDPowTwoThird)

# modelPlot(output_path_3_5*"model_run.jld2", output_path_3_5*experiment_name*"-fig", 
#     layers, (-25,25), (-25,25), nameattr= false, levels=10, localscale=false)
