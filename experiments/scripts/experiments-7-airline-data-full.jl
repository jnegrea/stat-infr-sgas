 using LinearAlgebra
# glm_path = "GLM_v2/"
# include(glm_path*"GLM_v2.jl")
source_path = "../core/"
include(source_path*"sgld-module-0.jl")
# include(source_path*"sgld-module-0-localglm.jl")

# using .SGLDModule

output_path = "../output/"
isdir(output_path) || mkdir(output_path)

output_path_7 = output_path*"experiments-7-airline-data-full/"
isdir(output_path_7) || mkdir(output_path_7)
include_str = true
post_fix = ""
if include_str
    post_fix *= "-all"
    #post_fix1 = "-uc-loglink"
    post_fix1 = "-loglink-poisson"
    #post_fix1 = "-uc-og-dt-custsqlink"
end


## Example 7.1: Airline Data for Paper
experiment_name = "experiments-7-1-airline-full-data-paper"
output_path_7_1 = output_path_7*experiment_name*"/"
isdir(output_path_7_1) || mkdir(output_path_7_1)

samplesize = 150000
w=1/2
bs = 250

algos = [ :preconSGD, :preconSGLD]
batchsizes = Dict(

    :preconSGD => bs,
    :preconSGLD => bs
    )
    
numstepsfactor = 1000


numstepss = Dict(
    :preconSGD => Int64(numstepsfactor*samplesize ÷ batchsizes[:preconSGD]),
    :preconSGLD => Int64(numstepsfactor*samplesize ÷ batchsizes[:preconSGLD])
)

keepperiods = Dict(
    :preconSGD => 1,
    :preconSGLD => 1
)
output_path_7_1_1= output_path_7_1*"/univariate-all/"
isdir(output_path_7_1_1) || mkdir(output_path_7_1_1)
# 
# modelRun(output_path_7*"airline-data-full-init-samplesize-150000-loglink-poisson.jld2", output_path_7_1*"model_run"*"_bs_"*string(bs)*post_fix1*".jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true, ismmap_run=false, keepperiods=keepperiods, fnameprefix=output_path_7_1)
# 
# modelSummary(output_path_7*"airline-data-full-init-samplesize-150000-loglink-poisson.jld2", output_path_7_1*"model_run"*"_bs_"*string(bs)*post_fix1*".jld2",
#     output_path_7_1*experiment_name*"_bs_"*string(bs)*post_fix1*"-summary.txt")

layers = layerer([:preconSGLD, :mixture, :allparms, :univariate, :histogram, :noaxes,:preconSGD, :sandwich,:trueparm])
modelPlot(output_path_7_1*"model_run"*"_bs_"*string(bs)*post_fix1*".jld2", output_path_7_1*"/univariate-all/"*experiment_name*"-ita-univ-fig-fontscaled"*post_fix1,#output_path_7_1*"/univariate-all-nolegend/"*experiment_name*"-univ-fig-fontscaled",
    layers, (-160,160), (-160,160), nameattr= false, levels=5, localscale=false, fontscale=2)

layers = layerer([:preconSGLD, :mixture, :allparms, :univariate, :histogram, :noaxes,:preconSGD, :sandwich,:trueparm, :nolegend])
modelPlot(output_path_7_1*"model_run"*"_bs_"*string(bs)*post_fix1*".jld2", output_path_7_1*"/univariate-all/"*experiment_name*"-ita-univ-fig-fontscaled-nolegend"*post_fix1,#output_path_7_1*"/univariate-all-nolegend/"*experiment_name*"-univ-fig-fontscaled",
    layers, (-160,160), (-160,160), nameattr= false, levels=5, localscale=false, fontscale=2)
