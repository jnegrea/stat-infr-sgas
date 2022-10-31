using JLD2
using StatsBase
using Statistics
using GLM
using DataFrames
using RData
using Distributions
using Plots
using LinearAlgebra

output_path = "../output/"
isdir(output_path) || mkdir(output_path)

source_data_file = output_path*"experiments-6-airline-data-full/airline-data-full-df.jld2"
## experiment-3 Logistic REgression Experiments

output_path_6 = output_path*"experiments-6-airline-data-full/"
isdir(output_path_6) || mkdir(output_path_6)


function airlineInit_full_nb(samplesize, fname, source_data_df_file, trueparmfile, modelfile, initparm=false)
    # source_data = load(source_data_file)
    @load source_data_df_file source_data_df

    N = (size(source_data_df)[1])
    # parmdim = 6
    
    @load trueparmfile parmdim Jinv V sqrtJinv trueparm 
    @load modelfile model_formula data_x_fun linkfun


    sample_indexes = sample(1:N, samplesize; replace=false, ordered=false)
    sample_data = source_data_df[sample_indexes,:]
    # sample_data_x = [ones(samplesize) Matrix(sample_data[:,["V3", "V4", "V5"]])]
    # sample_data_y = Vector(sample_data[:,"V2"])

    source_data = nothing
    GC.gc()

    mleparm = coef(glm(model_formula, sample_data, Poisson(), linkfun))
    
    data_x = data_x_fun(sample_data, samplesize)
    data_y = Vector(sample_data.ArrDelay .-15)
    
    data = [(Vector(data_x[i,:]),data_y[i]) for i in 1:samplesize]


    # gradloglik = (parm,xy) -> xy[1]*(xy[2] - logit(xy[1]'*parm ))
    # gradlogprior = parm -> zeros(parmdim)
    
    gradloglik = :poisson
    gradlogprior = :flat

    gradloglik_description = "(β,(x,y)) -> x*(y - exp(x'*β ))"
    gradlogprior_description = "β -> zeros(parmdim)"

    ismmap_init = false

    initparm = mleparm

    data_description = "Airline Dataset, Prepocessed by JN"
    
    @save fname parmdim Jinv V sqrtJinv data trueparm mleparm initparm gradloglik gradlogprior data_description samplesize ismmap_init
end

samplesize = 150000
fname = output_path_6*"airline-data-full-init.jld2"
source_data_df_file = output_path_6*"airline-data-full-df.jld2"
trueparmfile = output_path_6*"airline-data-full-trueparm.jld2"
modelfile = output_path_6*"airline-data-full-model.jld2"


airlineInit_full_nb(samplesize, fname, source_data_df_file, trueparmfile, modelfile)
