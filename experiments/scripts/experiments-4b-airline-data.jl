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

# layers = layerer([:halfsandwich, :sandwich])
# layers = layerer(layers, :vanillaSGD)
# layers = layerer(layers, :trueparm)
# layers = layerer(layers, :preconSGD; drop=:vanillaSGD)
# layers = layerer(layers, [:preconSGLD, :mixture, :jinv]; drops=[:preconSGD, :halfsandwich])

# airlineInit(samplesize, output_path_4_1*"model_init", source_data_file; trueparmfile=output_path_4*"airline-data-trueparm.jld2" , initparm=false, mc_size=false, mleparmtype=:glm)
# modelRun(output_path_4_1*"model_init.jld2", output_path_4_1*"model_run",
#     algos, samplesize, batchsizes, w, numstepss; multithread=false, ismmap_run=true, keepperiods=keepperiods)
# modelPlot(output_path_4_1*"model_run.jld2", output_path_4_1*experiment_name*"-fig",
#     layers, (-25,25), (-25,25), nameattr= false, levels=10, localscale=false)
# modelSummary(output_path_4_1*"model_init.jld2", output_path_4_1*"model_run.jld2",
#     output_path_4_1*experiment_name*"-summary.txt")

## Functions for example 4.2
function scatter_plot(x, y, xlabel, ylabel; xlims = nothing, post_fix1 = "", picname = "", logscale = false, prior = :flat)
    if logscale
        p = Plots.scatter(x, y, colour = [:blue],markersize = 12,
            markershape = [:circle],markeralpha = 0.8,label = nothing, xtickfontsize=18, xlims = xlims,
            ytickfontsize=18, xlabel = xlabel, ylabel = ylabel,labelfontsize = 18, yaxis = :log, xaxis = :log)
    else
        p = Plots.scatter(x, y, colour = [:blue],markersize = 12,
            markershape = [:circle],markeralpha = 0.8,label = nothing, xtickfontsize=18, xlims = xlims,
            ytickfontsize=18, xlabel = xlabel, ylabel = ylabel,labelfontsize = 18)
    end
    Plots.abline!(1, 0, line=:dash, subplot = 1,label = false)

    if !isempty(picname)
        if xlims!=nothing
            picname *= "-xlim"
        end
        if logscale
            picname*= "-logscale"
        end
        prior = string(prior)
        savefig(p, picname*"-"*String(prior)*post_fix1*".pdf")
    end
    return p
end

function equation_9_precon(Qinf, Jinv, V, Precon; ch = 4, cb = 1, cbBar = 1)
    B = ch.*Precon*inv(Jinv)
    right = ch^(2)*cbBar/(4*cb) .* Precon * V * Precon'
    left = 0.5.*(B*Qinf + Qinf'*B')#transpose(B))
    return left, right
end

function equation_12_precon(covItAvg, Jinv, V, Qinf, samplesize, iteravg_numepochs, precon; ch = 4, cb = 1)
    left = samplesize*iteravg_numepochs.* covItAvg
    J = inv(Jinv)
    sym1 = 0.5 .* (inv(precon*J) * Qinf + Qinf' * (inv(precon*J))')#* inv(J'*Precon'))

    exp_expr = -1*ch*iteravg_numepochs.* precon*J./(2*cb)
    expr = inv(precon*J)^2 * (Matrix{Float64}(I, size(Jinv)) - exp(exp_expr)) * Qinf
    sym2 = 0.5 .* (expr + expr')#transpose(Jinv))
    right = (4*cb/ch).*sym1 - (8*cb^2)/ch^2 /iteravg_numepochs .* sym2
    println("The constants for the first term ", 4*cb/ch)
    println("The constants for the second term ", (8*cb^(2))/ch^(2))
    return left, right
end


## Example 4.2: Medium Test Version of Airline Data
experiment_name = "experiments-4-2-airline-data-medium-test"
output_path_4_2 = output_path_4*experiment_name*"/"
isdir(output_path_4_2) || mkdir(output_path_4_2)
post_fix1 = ""
# samplesize = 100000
samplesize = 10000
w=1/2
#
algos = [:preconSGDdiagPowOneSumOne, :preconSGDdiagPowTwoThirdSumOne, :preconSGDdiagPowHalfSumOne, :preconSGDdiagPowThirdSumOne]
pows = [1, 2/3, 1/2, 1/3]

iteravg_numepochs=1 # NEW
num_iteravgs = 1000
numepochs = num_iteravgs*iteravg_numepochs

#algos = [:vanillaSGD, :preconSGD, :preconSGLD]

# batchsizes = Dict(
#     :vanillaSGD => 100,
#     :preconSGD => 100,
#     :preconSGLD => 100)
# batchsizes = Dict(
#     :preconSGDdiag => 1)
batchsizes = Dict(
    :preconSGDdiagPowOneSumOne => 1*ceil(Int,samplesize^(1-1)),
    :preconSGDdiagPowTwoThirdSumOne => 1*ceil(Int,samplesize^(1-2/3)),
    :preconSGDdiagPowHalfSumOne => 1*ceil(Int,samplesize^(1-1/2)),
    :preconSGDdiagPowThirdSumOne => 1*ceil(Int,samplesize^(1-1/3))
)

# numstepss = Dict(
#     algo => numepochs*samplesize ÷ batchsizes[algo] for algo in algos
# )

#numstepsfactor = 1000
# numstepss = Dict(
#     :vanillaSGD => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGD],
#     :preconSGD => numstepsfactor*samplesize ÷ batchsizes[:preconSGD],
#     :preconSGLD => numstepsfactor*samplesize ÷ batchsizes[:preconSGLD])
# numstepss = Dict(
#     :preconSGDdiag => numstepsfactor*samplesize ÷ batchsizes[:preconSGDdiag])
numstepss = Dict(
    algo => numepochs*samplesize ÷ batchsizes[algo] for algo in algos
)
# keepperiods = Dict(
#     :vanillaSGD => samplesize ÷ batchsizes[:vanillaSGD],
#     :preconSGD => samplesize ÷ batchsizes[:preconSGD],
#     :preconSGLD => samplesize ÷ batchsizes[:preconSGLD])
keepperiods = Dict(
    :preconSGDdiagPowOneSumOne => 1,
    :preconSGDdiagPowTwoThirdSumOne => 1,
    :preconSGDdiagPowHalfSumOne => 1,
    :preconSGDdiagPowThirdSumOne => 1
)

# layers = layerer([:halfsandwich, :sandwich])
# layers = layerer(layers, :vanillaSGD)
# layers = layerer(layers, :trueparm)
# layers = layerer(layers, :preconSGD; drop=:vanillaSGD)
# layers = layerer(layers, [:preconSGLD, :mixture, :jinv]; drops=[:preconSGD, :halfsandwich])

airlineInit(samplesize, output_path_4_2*"model_init", source_data_file; trueparmfile=output_path_4*"airline-data-trueparm.jld2" , initparm=false, mc_size=false, mleparmtype=:glm)
modelRun(output_path_4_2*"model_init.jld2", output_path_4_2*"model_run"*".jld2",
    algos, samplesize, batchsizes, w, numstepss; multithread = true, ismmap_run=false, keepperiods=keepperiods)
# modelPlot(output_path_4_2*"model_run.jld2", output_path_4_2*experiment_name*"-fig",
#     layers, (-25,25), (-25,25), nameattr= false, levels=10, localscale=false)
modelSummary(output_path_4_2*"model_init.jld2", output_path_4_2*"model_run.jld2",
    output_path_4_2*experiment_name*"-summary.txt")

# layers = layerer([:halfsandwich, :sandwich, :univariate])
# layers = layerer(layers, :vanillaSGD)
# layers = layerer(layers, :trueparm)
# layers = layerer(layers, :preconSGD)
# layers = layerer(layers, [:preconSGLD, :mixture, :jinv])
#
# modelPlot(output_path_4_2*"model_run.jld2", output_path_4_2*experiment_name*"-univariate-fig",
#     layers, (-25,25), (-25,25), nameattr= false, levels=10, localscale=false)
## Example 4.2: Iterative average test
modelrunfile = output_path_4_2*"model_run"*".jld2"
modelinitfile = output_path_4_2*"model_init.jld2"
#modelinitfile = "airline-data-full-nb-init-samplesize-150000-uc-logitlink-binom-whiteningSigmaX.jld2"
@load modelinitfile gradlogprior
@load modelrunfile algos parms trueparm mleparm Jinv V sqrtJinv w samplesize batchsizes numstepss ismmap_init ismmap_run keepperiods
# iteravg_numepochs = 16


samplesize
parms
# num_iteravgs = 2000
J = inv(Jinv)
pardim = length(trueparm)
w
#iteravg_numepochs = 1
sandwich = Jinv'*V*Jinv/samplesize/iteravg_numepochs
#mixture = (w*Jinv + (1-w)*Jinv'*V*Jinv)/samplesize#/
covsSandwich = sandwich[tril!(trues(size(sandwich)))]
varsSandwich = diag(sandwich)
stdsSandwich = sqrt.(diag(sandwich))

parms
local_mleparm = mleparm
localmap = p -> p

local_parms = Dict(algo => localmap.(parm) for (algo,parm) in parms)
batchsizes
samplesize
epochlengths = Dict(algo => samplesize ÷ batchsizes[algo] ÷ keepperiods[algo] for algo in algos)
numepochs = Dict(algo => numstepss[algo]*batchsizes[algo] ÷ samplesize for algo in algos)

iteravg_lengths = Dict(algo => epochlengths[algo]*iteravg_numepochs for algo in algos)

numiteravgs = Dict(algo => numstepss[algo] ÷ iteravg_lengths[algo] for algo in algos)
# iteravgranges = Dict(algo => [((2*i-1)*(numstepss[algo]÷ keepperiods[algo]÷100)+1):(min((2*i)*(numstepss[algo]÷ keepperiods[algo]÷100), (2*i-1)*(numstepss[algo]÷ keepperiods[algo]÷100)+1*epochlengths[algo])) for i in 1:50] for algo in algos)
iteravgranges = Dict(algo => [((i-1)*iteravg_lengths[algo]+1):(i*iteravg_lengths[algo]) for i in 1:numiteravgs[algo]] for algo in algos)
# print(iteravgranges)

iteravgs = Dict(algo => [sum(local_parm[range])/length(range) for range in iteravgranges[algo]] for (algo,local_parm) in local_parms)

covItAvgs = Dict(algo => cov(iteravg, corrected = true) for (algo, iteravg) in iteravgs)
qInfs = Dict(algo => cov(parm, corrected = true).*samplesize for (algo,parm) in parms)
# iteravgHalf = iteravgs[:vanillaSGDPowHalfSumOne]
# iteravgThird = iteravgs[:vanillaSGDPowThirdSumOne]
meanItAvgs = Dict(algo => mean(iteravg) for (algo, iteravg) in iteravgs)
stdItAvgs = Dict(algo => sqrt.(diag(covItAvg)) for (algo, covItAvg) in covItAvgs)
covItAvgsToPlot = Dict(algo => covItAvg[tril!(trues(size(covItAvg)))] for (algo, covItAvg) in covItAvgs)

eigenThird = eigen(sandwich - covItAvgs[:])
evaluesThird = eigenThird.values
print(evaluesThird)
ylabel = Dict(:mean => "MLE", :cov => "Sandwich Estimator")
xCovsLabels = Dict(:vanillaSGDPowHalfSumOne => "covariances of vanillaSGDPowHalfSumOne", :vanillaSGDPowThirdSumOne => "covariances of vanillaSGDPowThirdSumOne")


for algo in algos
    scatter_plot(covsSandwich, covItAvgsToPlot[algo], ylabel[:cov], "predicted covariance"; xlims = nothing, post_fix1 = post_fix1, picname = output_path_4_2*"/"*"covs_"*string(algo)*"_iteravg_numepochs_"*string(iteravg_numepochs)*"_batchsize_"*string(batchsizes[algo])*"_samplesize_"*string(samplesize), prior = gradlogprior)
end



ch = 4
precon = inv(Diagonal(inv(Jinv)))
#Precon = Jinv
B = ch.*precon*inv(Jinv)
E = eigen(B)
lambdaMin = findmin(E.values)
1/lambdaMin[1]

for algo in algos
    # SymQB, A = equation_9_precon(qInfs[algo], Jinv, V, precon)
    # SymQBtoPlot = SymQB[tril!(trues(size(SymQB)))]
    # AtoPlot = A[tril!(trues(size(A)))]
    # scatter_plot(AtoPlot, SymQBtoPlot, "right-handsided equation", "left-handsided equation"; xlims = nothing, post_fix1 = post_fix1, picname = output_path_4_2*"eq9_"*String(algo)*"_batchsize_"*string(batchsizes[algo])*"_samplesize_"*string(samplesize), prior = string(gradlogprior))
    #
    # SymQBDiag = diag(SymQB)
    # ADiag = diag(A)
    # scatter_plot(ADiag, SymQBDiag, "right-handsided equation", "left-handsided equation"; xlims = nothing, post_fix1 = post_fix1, picname = output_path_4_2*"eq9_diag_"*String(algo)*"_batchsize_"*string(batchsizes[algo])*"_samplesize_"*string(samplesize), prior = string(gradlogprior))
    #
    # SymQBOffDiag = SymQB[tril!(trues(size(SymQB)), -1)]
    # AOffDiag = A[tril!(trues(size(A)), -1)]
    # scatter_plot(AOffDiag, SymQBOffDiag, "right-handsided equation", "left-handsided equation"; xlims = nothing, post_fix1 = post_fix1, picname = output_path_4_2 *"eq9_off_diag_"*String(algo)*"_batchsize_"*string(batchsizes[algo])*"_samplesize_"*string(samplesize), prior = string(gradlogprior))

    left, right = equation_12_precon(covItAvgs[algo], Jinv, V, qInfs[algo], samplesize, iteravg_numepochs, precon)
    leftToPlot = left[tril!(trues(size(left)))]
    rightToPlot = right[tril!(trues(size(right)))]
    scatter_plot(rightToPlot, leftToPlot, "predicted covariance", "empirical covariance"; xlims = nothing, post_fix1 = post_fix1, picname = output_path_4_2 *"eq12_"*String(algo)*"_iteravg_numepochs_"*string(iteravg_numepochs)*"_batchsize_"*string(batchsizes[algo])*"_samplesize_"*string(samplesize), prior = string(gradlogprior))#, logscale = true)

    DiagLeft = diag(left)
    DiagRight = diag(right)
    scatter_plot(DiagRight, DiagLeft, "predicted covariance", "empirical covariance"; xlims = nothing, post_fix1 = post_fix1, picname = output_path_4_2 *"eq12_diag_"*String(algo)*"_iteravg_numepochs_"*string(iteravg_numepochs)*"_batchsize_"*string(batchsizes[algo])*"_samplesize_"*string(samplesize), prior = string(gradlogprior))#, logscale = true)
end
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
# modelRun(output_path_4_3*"model_init.jld2", output_path_4_3*"model_run",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true, ismmap_run=true, keepperiods=keepperiods)
# modelPlot(output_path_4_3*"model_run.jld2", output_path_4_3*experiment_name*"-fig",
#     layers, (-30,30), (-30,30), nameattr= false, levels=10, localscale=false)
# modelSummary(output_path_4_3*"model_init.jld2", output_path_4_3*"model_run.jld2",
#     output_path_4_3*experiment_name*"-summary.txt")
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
# modelRun(output_path_4_4*"model_init.jld2", output_path_4_4*"model_run",
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
# modelPlot(output_path_4_5*"model_run.jld2", output_path_4_5*"/univariate-all/"*experiment_name*"-fontscaled-univ-fig",
#     layers, (-60,60), (-60,60), nameattr= false, levels=5, localscale=false, fontscale=2.2)
# modelSummary(output_path_4_5*"model_init.jld2", output_path_4_5*"model_run.jld2",
#     output_path_4_5*experiment_name*"-summary.txt")

layers = layerer([:vanillaSGLD, :jinv, :trueparm, :univariate, :histogram,:allparms, :nolegend])
# layers = layerer(layers, [:vanillaSGLDdecreasing], drop=[:vanillaSGLD])
layers = layerer(layers, [:preconSGD])
layers = layerer(layers, [:preconVSGD], drop=[:preconSGD])
layers = layerer(layers, [:preconSGLD], drop=[:preconVSGD])
# modelPlot(output_path_4_5*"model_run.jld2", output_path_4_5*"/univariate-all-nolegend/"*experiment_name*"-fontscaled-univ-fig",
#     layers, (-60,60), (-60,60), nameattr= false, levels=5, localscale=false, fontscale=2.2)


## Example 4.6: Iteravg test
experiment_name = "experiments-4-6-airline-data-iteravg-test"
output_path_4_6 = output_path_4*experiment_name*"/"
isdir(output_path_4_6) || mkdir(output_path_4_6)

samplesize = 1000000
w=1/2


algos = [:vanillaSGDPowHalf, :vanillaSGDPowTwoThird, :vanillaSGDPowThird]

batchsizes = Dict(
    :vanillaSGDPowHalf => 1000,
    :vanillaSGDPowTwoThird => 1000,
    :vanillaSGDPowThird => 1000)

numstepsfactor = 10000
numstepss = Dict(
    :vanillaSGDPowHalf => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGDPowHalf],
    :vanillaSGDPowTwoThird => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGDPowTwoThird],
    :vanillaSGDPowThird => numstepsfactor*samplesize ÷ batchsizes[:vanillaSGDPowThird])

keepperiods = Dict(
    :vanillaSGDPowHalf => 1,
    :vanillaSGDPowTwoThird => 1,
    :vanillaSGDPowThird => 1)

# airlineInit(samplesize, output_path_4_6*"model_init", source_data_file; trueparmfile=output_path_4*"airline-data-trueparm.jld2" , initparm=false, mc_size=false, mleparmtype=:glm)
# modelRun(output_path_4_6*"model_init.jld2", output_path_4_6*"model_run.jld2",
#     algos, samplesize, batchsizes, w, numstepss; multithread=true, ismmap_run=true, keepperiods=keepperiods)
#
# layers = layerer(:sandwich)
# layers = layerer(layers, [:vanillaSGDPowHalf, :iteravg])
# layers = layerer(layers, :trueparm)
# layers = layerer(layers, :vanillaSGDPowTwoThird; drop=:vanillaSGDPowHalf)
# layers = layerer(layers, :vanillaSGDPowThird; drop=:vanillaSGDPowTwoThird)
#
# modelPlot(output_path_4_6*"model_run.jld2", output_path_4_6*experiment_name*"-fig",
#     layers, (-25,25), (-25,25), nameattr= false, levels=10, localscale=false)
