using LinearAlgebra

source_path = "../core/"
include(source_path*"sgld-module-0.jl")
# using .SGLDModule

output_path = "../output/"
isdir(output_path) || mkdir(output_path)

output_path_6 = output_path*"experiments-6-airline-data-full/"
isdir(output_path_6) || mkdir(output_path_6)


## Example 6.1: Airline Data for Paper
experiment_name = "experiments-6-1-airline-full-data-paper"
output_path_6_1 = output_path_6*experiment_name*"/"
isdir(output_path_6_1) || mkdir(output_path_6_1)

samplesize = 150000
w=1/2


# algos = [:vanillaSGD, :preconSGD, :preconVSGD, :preconSGLD, :vanillaSGLD, :vanillaSGLDdecreasing]

algos = [:preconSGD, :preconVSGD, :preconSGLD]

batchsizes = Dict(
    # :vanillaSGD => 10,
    :preconSGD => 1000,
    :preconVSGD => 1000,
    :preconSGLD => 1000,
    # :vanillaSGLD => 10,
    # :vanillaSGLDdecreasing => 10
    )

numstepsfactor = 1000

numstepss = Dict(
    # :vanillaSGD => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGD], 
    :preconSGD => numstepsfactor*samplesize ÷ batchsizes[:preconSGD],
    :preconVSGD => numstepsfactor*samplesize ÷ batchsizes[:preconVSGD],
    :preconSGLD => numstepsfactor*samplesize ÷ batchsizes[:preconSGLD],
    # :vanillaSGLD =>  numstepsfactor*samplesize ÷ batchsizes[:vanillaSGD],
    # :vanillaSGLDdecreasing =>  numstepsfactor*samplesize ÷ batchsizes[:vanillaSGLDdecreasing]
)

keepperiods = Dict(
    # :vanillaSGD => 1, 
    :preconSGD => 1,
    :preconVSGD => 1,
    :preconSGLD => 1,
    # :vanillaSGLD => 1,
    # :vanillaSGLDdecreasing => 1
)
    

# layers = layerer([:vanillaSGLD, :jinv, :trueparm])
layers = layerer([:preconSGD, :jinv, :trueparm, :nogrid])
# layers = layerer(layers, [:vanillaSGLDdecreasing], drop=[:vanillaSGLD])
# layers = layerer(layers, [:preconSGD], drop=[:vanillaSGLD])
layers = layerer(layers, [:preconVSGD], drop=[:preconSGD])
layers = layerer(layers, [:preconSGLD], drop=[:preconVSGD])

# 
# modelRun(output_path_6*"airline-data-full-init.jld2", output_path_6_1*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true, ismmap_run=true, keepperiods=keepperiods, fnameprefix=output_path_6_1)

# modelPlot(output_path_6_1*"model_run.jld2", output_path_6_1*experiment_name*"-fig-fontscaled", 
    # layers, (-12,12), (-.005,.005), nameattr= false, levels=5, localscale=false, fontscale=1.5)

# layers = layerer([:vanillaSGLD, :jinv, :trueparm, :univariate, :histogram,:allparms])
layers = layerer([:preconVSGD, :jinv, :trueparm, :univariate, :histogram, :allparms])
# layers = layerer(layers, [:vanillaSGLDdecreasing], drop=[:vanillaSGLD])
layers = layerer(layers, [:preconSGD, :sandwich], drop=[:preconVSGD, :jinv])
layers = layerer(layers, [:preconSGLD, :mixture], drop=[:preconSGD, :sandwich])
# layers = layerer(layers, [:preconSGLD], drop=[:preconVSGD])
modelPlot(output_path_6_1*"model_run.jld2", output_path_6_1*"/univariate-all/"*experiment_name*"-univ-fig-fontscaled", 
    layers, (-60,60), (-60,60), nameattr= false, levels=5, localscale=false, fontscale=2)
    
# modelSummary(output_path_6*"airline-data-full-init.jld2", output_path_6_1*"model_run.jld2", 
#     output_path_6_1*experiment_name*"-summary.txt")
# 

# output_path_6_1_1= output_path_6_1*"/univariate-all/"
# layers = layerer([:preconSGLD, :mixture, :allparms, :univariate, :histogram, :nolegend])
# layers = layerer(layers, [:preconSGD, :sandwich])
# # layers = layerer(layers, :trueparm)
# # layers = layerer(layers, [:preconSGLD, :mixture])
# layers = layerer(layers, [:preconVSGD, :jinv])


layers = layerer([:preconSGLD, :mixture, :allparms, :univariate, :histogram, :nolegend])
layers = layerer(layers, [:preconSGD, :sandwich])
# layers = layerer(layers, :trueparm)
# layers = layerer(layers, [:preconSGLD, :mixture])
layers = layerer(layers, [:preconVSGD, :jinv])
modelPlot(output_path_6_1*"model_run.jld2", output_path_6_1*"/univariate-all-nolegend/"*experiment_name*"-univ-fig-fontscaled", 
    layers, (-60,60), (-60,60), nameattr= false, levels=5, localscale=false, fontscale=2)
    

# modelPlot(output_path_6_1*"model_run.jld2", output_path_6_1_1*"yourmodeliswrong/"*experiment_name*"-nolegend-fig", 
#     layers, (-60,60), (-60,60), nameattr= false, levels = 6, localscale=false)    
# modelSummary(output_path_6*"airline-data-full-init.jld2", output_path_6_1*"model_run.jld2", 
#     output_path_6_1_1*"yourmodeliswrong/"*experiment_name*"-summary.txt")    

layers = layerer([:jinv, :sandwich, :allparms, :univariate, :histogram, :nolegend])
layers = layerer(layers, [:preconSGLD, :mixture])
layers = layerer(layers, [:preconSGD])
# layers = layerer(layers, :trueparm)
# layers = layerer(layers, [:preconSGLD, :mixture])
layers = layerer(layers, [:preconVSGD])

# modelPlot(output_path_6_1*"model_run.jld2", output_path_6_1_1*"yourmodeliswrong/"*experiment_name*"-nolegend-fig", 
#     layers, (-60,60), (-60,60), nameattr= false, levels = 6, localscale=false)    
# modelSummary(output_path_6*"airline-data-full-init.jld2", output_path_6_1*"model_run.jld2", 
#     output_path_6_1_1*"yourmodeliswrong/"*experiment_name*"-summary.txt")    
