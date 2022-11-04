using LinearAlgebra

source_path = "../core/"
include(source_path*"sgld-module-0.jl")
# using .SGLDModule

output_path = "../output/"
isdir(output_path) || mkdir(output_path)

source_data_file = "../source-data/data_airline_raw.rdata"
## experiment-3 Logistic REgression Experiments

output_path_4 = output_path*"experiments-4-airline-data/"
isdir(output_path_4) || mkdir(output_path_4)

## Example 4.1: Small Test Version of Airline Data
experiment_name = "experiments-4-1-airline-data-small-test"
output_path_4_1 = output_path_4*experiment_name*"/"
isdir(output_path_4_1) || mkdir(output_path_4_1)

samplesize = 100
w=1/2


algos = [:vanillaSGD, :preconSGD, :preconSGLD]

batchsizes = Dict(
    :vanillaSGD => 10,
    :preconSGD => 10,
    :preconSGLD => 10)

numstepsfactor = 10
numstepss = Dict(
    :vanillaSGD => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGD],
    :preconSGD => numstepsfactor*samplesize ÷ batchsizes[:preconSGD],
    :preconSGLD => numstepsfactor*samplesize ÷ batchsizes[:preconSGLD])

keepperiods = Dict(
    :vanillaSGD => samplesize ÷ batchsizes[:vanillaSGD],
    :preconSGD => samplesize ÷ batchsizes[:preconSGD],
    :preconSGLD => samplesize ÷ batchsizes[:preconSGLD])

layers = layerer([:halfsandwich, :sandwich])
layers = layerer(layers, :vanillaSGD)
layers = layerer(layers, :trueparm)
layers = layerer(layers, :preconSGD; drop=:vanillaSGD)
layers = layerer(layers, [:preconSGLD, :mixture, :jinv]; drops=[:preconSGD, :halfsandwich])

# airlineInit(samplesize, output_path_4_1*"model_init", source_data_file; trueparmfile=output_path_4*"airline-data-trueparm.jld2" , initparm=false, mc_size=false, mleparmtype=:glm)
# modelRun(output_path_4_1*"model_init.jld2", output_path_4_1*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=false, ismmap_run=true, keepperiods=keepperiods)
# modelPlot(output_path_4_1*"model_run.jld2", output_path_4_1*experiment_name*"-fig",
#     layers, (-25,25), (-25,25), nameattr= false, levels=10, localscale=false)
# modelSummary(output_path_4_1*"model_init.jld2", output_path_4_1*"model_run.jld2",
#     output_path_4_1*experiment_name*"-summary.txt")


## Example 4.2: Medium Test Version of Airline Data
experiment_name = "experiments-4-2-airline-data-medium-test"
output_path_4_2 = output_path_4*experiment_name*"/"
isdir(output_path_4_2) || mkdir(output_path_4_2)

samplesize = 100000
w=1/2


algos = [:vanillaSGD, :preconSGD, :preconSGLD]

batchsizes = Dict(
    :vanillaSGD => 100,
    :preconSGD => 100,
    :preconSGLD => 100)

numstepsfactor = 1000
numstepss = Dict(
    :vanillaSGD => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGD],
    :preconSGD => numstepsfactor*samplesize ÷ batchsizes[:preconSGD],
    :preconSGLD => numstepsfactor*samplesize ÷ batchsizes[:preconSGLD])

keepperiods = Dict(
    :vanillaSGD => samplesize ÷ batchsizes[:vanillaSGD],
    :preconSGD => samplesize ÷ batchsizes[:preconSGD],
    :preconSGLD => samplesize ÷ batchsizes[:preconSGLD])

layers = layerer([:halfsandwich, :sandwich])
layers = layerer(layers, :vanillaSGD)
layers = layerer(layers, :trueparm)
layers = layerer(layers, :preconSGD; drop=:vanillaSGD)
layers = layerer(layers, [:preconSGLD, :mixture, :jinv]; drops=[:preconSGD, :halfsandwich])

# airlineInit(samplesize, output_path_4_2*"model_init", source_data_file; trueparmfile=output_path_4*"airline-data-trueparm.jld2" , initparm=false, mc_size=false, mleparmtype=:glm)
# modelRun(output_path_4_2*"model_init.jld2", output_path_4_2*"model_run.jld",
#     algos, samplesize, batchsizes, w, numstepss; multithread=false, ismmap_run=true, keepperiods=keepperiods)
# modelPlot(output_path_4_2*"model_run.jld2", output_path_4_2*experiment_name*"-fig",
#     layers, (-25,25), (-25,25), nameattr= false, levels=10, localscale=false)
# modelSummary(output_path_4_2*"model_init.jld2", output_path_4_2*"model_run.jld2",
#     output_path_4_2*experiment_name*"-summary.txt")

layers = layerer([:halfsandwich, :sandwich, :univariate])
layers = layerer(layers, :vanillaSGD)
layers = layerer(layers, :trueparm)
layers = layerer(layers, :preconSGD)
layers = layerer(layers, [:preconSGLD, :mixture, :jinv])

# modelPlot(output_path_4_2*"model_run.jld2", output_path_4_2*experiment_name*"-univariate-fig",
#     layers, (-25,25), (-25,25), nameattr= false, levels=10, localscale=false)
#
## Example 4.3: Large Test Version of Airline Data
experiment_name = "experiments-4-3-airline-data-large-test"
output_path_4_3 = output_path_4*experiment_name*"/"
isdir(output_path_4_3) || mkdir(output_path_4_3)

samplesize = Int64(1e6)
w=1/2


algos = [:vanillaSGD, :preconSGD, :preconSGLD]

batchsizes = Dict(
    :vanillaSGD => 1000,
    :preconSGD => 1000,
    :preconSGLD => 1000)

numstepsfactor = 1000
numstepss = Dict(
    :vanillaSGD => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGD],
    :preconSGD => numstepsfactor*samplesize ÷ batchsizes[:preconSGD],
    :preconSGLD => numstepsfactor*samplesize ÷ batchsizes[:preconSGLD])

keepperiods = Dict(
    :vanillaSGD => samplesize ÷ batchsizes[:vanillaSGD]÷10 ,
    :preconSGD => samplesize ÷ batchsizes[:preconSGD]÷10,
    :preconSGLD => samplesize ÷ batchsizes[:preconSGLD]÷10)

layers = layerer([:halfsandwich, :sandwich])
layers = layerer(layers, :vanillaSGD)
layers = layerer(layers, :trueparm)
layers = layerer(layers, :preconSGD; drop=:vanillaSGD)
layers = layerer(layers, [:preconSGLD, :mixture, :jinv]; drops=[:preconSGD, :halfsandwich])

# airlineInit(samplesize, output_path_4_3*"model_init", source_data_file; trueparmfile=output_path_4*"airline-data-trueparm.jld2" , initparm=false, mc_size=false, mleparmtype=:glm)
# modelRun(output_path_4_3*"model_init.jld2", output_path_4_3*"model_run.jld2",
    # algos, samplesize, batchsizes, w, numstepss; multithread=true, ismmap_run=true, keepperiods=keepperiods)
# modelPlot(output_path_4_3*"model_run.jld2", output_path_4_3*experiment_name*"-fig",
#     layers, (-30,30), (-30,30), nameattr= false, levels=10, localscale=false)
# modelSummary(output_path_4_3*"model_init.jld2", output_path_4_3*"model_run.jld2",
    # output_path_4_3*experiment_name*"-summary.txt")
layers = layerer([:halfsandwich, :sandwich, :univariate])
layers = layerer(layers, :vanillaSGD)
layers = layerer(layers, :trueparm)
layers = layerer(layers, :preconSGD)
layers = layerer(layers, [:preconSGLD, :mixture, :jinv])

# modelPlot(output_path_4_3*"model_run.jld2", output_path_4_3*experiment_name*"-univariate-fig",
#     layers, (-30,30), (-30,30), nameattr= false, levels=10, localscale=false)


## Example 4.4: Airline Data for Presentation
experiment_name = "experiments-4-4-airline-data-presentation"
output_path_4_4 = output_path_4*experiment_name*"/"
isdir(output_path_4_4) || mkdir(output_path_4_4)

samplesize = Int64(1e6)
w=1/2


algos = [:vanillaSGD, :vanillaSGLD, :preconSGD, :preconSGLD, :vanillaSGLDdecreasing, :largeBatchPreconSGLD]

batchsizes = Dict(
    :vanillaSGD => 1000,
    :vanillaSGLD => 1000,
    :preconSGD => 1000,
    :preconSGLD => 1000,
    :vanillaSGLDdecreasing => 1000,
    :largeBatchPreconSGLD => samplesize÷4
)

numstepsfactor = 100
numstepss = Dict(
    :vanillaSGD => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGD],
    :vanillaSGLD => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGLD],
    :preconSGD => numstepsfactor*samplesize ÷ batchsizes[:preconSGD],
    :preconSGLD => numstepsfactor*samplesize ÷ batchsizes[:preconSGLD],
    :vanillaSGLDdecreasing => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGLDdecreasing],
    :largeBatchPreconSGLD => numstepsfactor*samplesize ÷ batchsizes[:largeBatchPreconSGLD]
)

# keepperiods = Dict(
#     :vanillaSGD => samplesize ÷ batchsizes[:vanillaSGD]÷10,
#     :vanillaSGLD => samplesize ÷ batchsizes[:vanillaSGLD]÷10 ,
#     :preconSGD => samplesize ÷ batchsizes[:preconSGD]÷10,
#     :preconSGLD => samplesize ÷ batchsizes[:preconSGLD]÷10,
#     :vanillaSGLDdecreasing => samplesize ÷ batchsizes[:vanillaSGLDdecreasing]÷10,
#     :largeBatchPreconSGLD => max(1,samplesize ÷ batchsizes[:largeBatchPreconSGLD]÷10),
#     )

keepperiods = Dict(
    :vanillaSGD => 1,
    :vanillaSGLD => 1 ,
    :preconSGD => 1,
    :preconSGLD => 1,
    :vanillaSGLDdecreasing => 1,
    :largeBatchPreconSGLD => 1,
    )


layers = layerer([:sandwich, :jinv])
layers = layerer(layers, :vanillaSGLD)
# layers = layerer(layers, :trueparm)
layers = layerer(layers, [:preconSGLD]; drop=:vanillaSGLD)
# layers = layerer(layers, [:vanillaSGLDdecreasing]; drops=[:preconSGLD, :mixture])
# layers = layerer(layers, [:largeBatchPreconSGLD]; drops=[:vanillaSGLDdecreasing])

# airlineInit(samplesize, output_path_4_4*"model_init", source_data_file; trueparmfile=output_path_4*"airline-data-trueparm.jld2" , initparm=false, mc_size=false, mleparmtype=:glm)
# modelRun(output_path_4_4*"model_init.jld2", output_path_4_4*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true, ismmap_run=true, keepperiods=keepperiods, fnameprefix=output_path_4_4)
# modelPlot(output_path_4_4*"model_run.jld2", output_path_4_4*experiment_name*"-fig",
    # layers, (-60,60), (-60,60), nameattr= false, levels=10, localscale=false)
# modelSummary(output_path_4_4*"model_init.jld2", output_path_4_4*"model_run.jld2",
#     output_path_4_4*experiment_name*"-summary.txt")
layers = layerer([:sandwich, :jinv, :univariate, :histogram])
layers = layerer(layers, :vanillaSGLD)
# layers = layerer(layers, :trueparm)
layers = layerer(layers, [:preconSGLD])
# layers = layerer(layers, [:vanillaSGLDdecreasing])
# layers = layerer(layers, [:largeBatchPreconSGLD])


# modelPlot(output_path_4_4*"model_run.jld2", output_path_4_4*experiment_name*"-univariate-fig",
    # layers, (-60,60), (-60,60), nameattr= false, levels=10, localscale=false)

layers = layerer([:sandwich, :jinv, :univariate, :nolegend, :histogram])
layers = layerer(layers, :vanillaSGLD)
# layers = layerer(layers, :trueparm)
layers = layerer(layers, [:preconSGLD])
# layers = layerer(layers, [:vanillaSGLDdecreasing])
# layers = layerer(layers, [:largeBatchPreconSGLD])


# modelPlot(output_path_4_4*"model_run.jld2", output_path_4_4*experiment_name*"-nolegend-univariate-fig",
    # layers, (-60,60), (-60,60), nameattr= false, levels=10, localscale=false)


output_path_4_4_1= output_path_4_4*"/all-parms-univariate/"
isdir(output_path_4_4_1) || mkdir(output_path_4_4_1)
layers = layerer([:sandwich, :jinv, :allparms, :univariate, :histogram])
layers = layerer(layers, :vanillaSGLD)
# layers = layerer(layers, :trueparm)
layers = layerer(layers, [:preconSGLD])
# modelPlot(output_path_4_4*"model_run.jld2", output_path_4_4_1*experiment_name*"-fig",
    # layers, (-60,60), (-60,60), nameattr= false, levels = 6, localscale=false)

layers = layerer([:sandwich, :jinv, :allparms, :univariate, :histogram, :nolegend])
layers = layerer(layers, :vanillaSGLD)
# layers = layerer(layers, :trueparm)
layers = layerer(layers, [:preconSGLD])
# modelPlot(output_path_4_4*"model_run.jld2", output_path_4_4_1*experiment_name*"-nolegend-fig",
#     layers, (-60,60), (-60,60), nameattr= false, levels = 6, localscale=false)



layers = layerer([:sandwich, :jinv, :allparms, :univariate, :histogram, :nolegend])
# layers = layerer(layers, :vanillaSGLDdecreasing)
layers = layerer(layers, :vanillaSGLD)
# layers = layerer(layers, :preconSGD)
# layers = layerer(layers, :trueparm)
layers = layerer(layers, :preconSGLD)
# modelPlot(output_path_4_4*"model_run.jld2", output_path_4_4_1*"yourmodeliswrong/"*experiment_name*"-nolegend-fig",
    # layers, (-60,60), (-60,60), nameattr= false, levels = 6, localscale=false)
# modelSummary(output_path_4_4*"model_init.jld2", output_path_4_4*"model_run.jld2",
#     output_path_4_4_1*"yourmodeliswrong/"*experiment_name*"-summary.txt")

## Example 4.5: Airline Data for Paper
experiment_name = "experiments-4-5-airline-data-paper"
output_path_4_5 = output_path_4*experiment_name*"/"
isdir(output_path_4_5) || mkdir(output_path_4_5)

samplesize = Int64(1e6)
w=1/2


algos = [:vanillaSGD, :preconSGD, :preconVSGD, :preconSGLD, :vanillaSGLD, :vanillaSGLDdecreasing]

batchsizes = Dict(
    :vanillaSGD => 1000,
    :preconSGD => 1000,
    :preconVSGD => 1000,
    :preconSGLD => 1000,
    :vanillaSGLD => 1000,
    :vanillaSGLDdecreasing => 1000
    )

numstepsfactor = 1000

numstepss = Dict(
    :vanillaSGD => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGD],
    :preconSGD => numstepsfactor*samplesize ÷ batchsizes[:preconSGD],
    :preconVSGD => numstepsfactor*samplesize ÷ batchsizes[:preconVSGD],
    :preconSGLD => numstepsfactor*samplesize ÷ batchsizes[:preconSGLD],
    :vanillaSGLD =>  numstepsfactor*samplesize ÷ batchsizes[:vanillaSGD],
    :vanillaSGLDdecreasing =>  numstepsfactor*samplesize ÷ batchsizes[:vanillaSGLDdecreasing]
)

keepperiods = Dict(
    :vanillaSGD => 1,
    :preconSGD => 1,
    :preconVSGD => 1,
    :preconSGLD => 1,
    :vanillaSGLD => 1,
    :vanillaSGLDdecreasing => 1
)


layers = layerer([:vanillaSGLD, :jinv, :trueparm, :noaxes])
# layers = layerer(layers, [:vanillaSGLDdecreasing], drop=[:vanillaSGLD])
layers = layerer(layers, [:preconSGD], drop=[:vanillaSGLD])
layers = layerer(layers, [:preconVSGD], drop=[:preconSGD])
layers = layerer(layers, [:preconSGLD], drop=[:preconVSGD])

# airlineInit(samplesize, output_path_4_5*"model_init.jld2", source_data_file; trueparmfile=output_path_4*"airline-data-trueparm.jld2" , initparm=false, mc_size=false, mleparmtype=:glm)
# modelRun(output_path_4_5*"model_init.jld2", output_path_4_5*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true, ismmap_run=true, keepperiods=keepperiods, fnameprefix=output_path_4_5)

# modelPlot(output_path_4_5*"model_run.jld2", output_path_4_5*experiment_name*"-fontscaled-fig",
#     layers, (-12,12), (-50,50), nameattr= false, levels=5, localscale=false, fontscale=2.2, legendloc=:bottomleft)

layers = layerer([:vanillaSGLD, :jinv, :trueparm, :univariate, :histogram,:allparms])
# layers = layerer(layers, [:vanillaSGLDdecreasing], drop=[:vanillaSGLD])
layers = layerer(layers, [:preconSGD])
layers = layerer(layers, [:preconVSGD], drop=[:preconSGD])
layers = layerer(layers, [:preconSGLD], drop=[:preconVSGD])
modelPlot(output_path_4_5*"model_run.jld2", output_path_4_5*"/univariate-all/"*experiment_name*"-fontscaled-univ-fig",
    layers, (-60,60), (-60,60), nameattr= false, levels=5, localscale=false, fontscale=2.2, legendloc=:topright)
# modelSummary(output_path_4_5*"model_init.jld2", output_path_4_5*"model_run.jld2",
#     output_path_4_5*experiment_name*"-summary.txt")

layers = layerer([:vanillaSGLD, :jinv, :trueparm, :univariate, :histogram,:allparms, :nolegend])
# layers = layerer(layers, [:vanillaSGLDdecreasing], drop=[:vanillaSGLD])
layers = layerer(layers, [:preconSGD])
layers = layerer(layers, [:preconVSGD], drop=[:preconSGD])
layers = layerer(layers, [:preconSGLD], drop=[:preconVSGD])
# modelPlot(output_path_4_5*"model_run.jld2", output_path_4_5*"/univariate-all-nolegend/"*experiment_name*"-fontscaled-univ-fig",
#     layers, (-60,60), (-60,60), nameattr= false, levels=5, localscale=false, fontscale=2.2)
