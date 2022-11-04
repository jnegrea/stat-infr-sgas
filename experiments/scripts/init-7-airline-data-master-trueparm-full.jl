using JLD2
using StatsBase
using Statistics

using DataFrames
using RData
using Distributions
using Plots
using LinearAlgebra
using GLM

output_path = "../output/"
isdir(output_path) || mkdir(output_path)
source_data_name = "airline-data-full-df"
#include_str = true
include_str = true
post_fix = ""
post_fix1 = "-loglink-poisson"

if include_str
    post_fix *= "-all"
    #post_fix1 = "-uc-og-dt-custsqlink"
    #post_fix1 = "-uc-loglink"
    post_fix1 = "-loglink-poisson"
end

source_data_file = output_path*"experiments-7-airline-data-full/"*source_data_name*post_fix*".jld2"
## experiment-3 Logistic REgression Experiments
output_path_7 = output_path*"experiments-7-airline-data-full/"
isdir(output_path_7) || mkdir(output_path_7)
# @load modelfile model_formula data_x_fun linkfun family
# linkfun
# family
# model_formula
function airlineInit_full(samplesize, fname, source_data_df_file, trueparmfile, modelfile, initparm=false)
    # source_data = load(source_data_file)
    @load source_data_df_file source_data_df include_str
    println("if include all the string columns ", include_str)
    N = (size(source_data_df)[1])
    println(size(source_data_df))
    # parmdim = 6

    @load trueparmfile parmdim Jinv V sqrtJinv trueparm name
    @load modelfile model_formula data_x_fun linkfun family
    println(name)
    println("True parameter dimensions are ", parmdim)
    sample_indexes = sample(1:N, samplesize; replace=false, ordered=false)
    println("number of samples ", samplesize)
    sample_data = source_data_df[sample_indexes,:]
    println("size of sample data ", size(sample_data))
    # if !isempty(name)
    # 
    #     tmp = copy(sample_data[:,name])
    #     println("The maximum of variance is ", var(tmp))
    #     println("The column type is ", eltype(sample_data[:,name]))
    #     sample_data[!,name] = convert.(Float64, sample_data[:,name])
    #     sample_data[:,name] = (tmp .- mean(tmp))./sqrt(var(tmp))
    # 
    #     println("Then maximum variance of scaled covariate ", var(sample_data[:,name]))
    # 
    #     tmp = nothing
    #     inds = nothing
    #     GC.gc
    # end

    # sample_data_x = [ones(samplesize) Matrix(sample_data[:,["V3", "V4", "V5"]])]
    # sample_data_y = Vector(sample_data[:,"V2"])

    # source_data = nothing
    # GC.gc()
    if include_str
        data_x = data_x_fun(model_formula, sample_data)
    else
        data_x = data_x_fun(sample_data, samplesize)
    end
    println("size of data ", size(data_x))
    data_y = Vector(sample_data.ArrDelay .- 15)



    mleparm = GLM.coef(GLM.glm(model_formula, sample_data, family, linkfun))
    println("MLE dimensions are ",length(mleparm))
    println("MLE are ",mleparm)


    #data_y = Vector(sample_data.ArrDelay .-15)

    data = [(Vector(data_x[i,:]),data_y[i]) for i in 1:samplesize]

    #data_x_frame = GLM.ModelFrame(model_formula, sample_data, contrasts = Dict( :UniqueCarrier => GLM.DummyCoding(), :Origin => GLM.DummyCoding(),:Dest => GLM.DummyCoding()))
    # data_x_frame = GLM.ModelFrame(model_formula, sample_data, contrasts = Dict( :UniqueCarrier => GLM.DummyCoding()))
    # x_names = coefnames(data_x_frame)
    # println("The covariate with largest eigenvalue is ", x_names[ind])
    # if length(mleparm)!=length(trueparm)
    # 
    #     shuffle_indices=sample(1:N,N; replace=false, ordered=false)
    #     source_data = source_data_df[shuffle_indices,:]
    #     # shuffle_indices=nothing
    #     # source_data_df = nothing
    #     # GC.gc()
    # 
    #     if include_str
    #         data_x_full = data_x_fun(model_formula, source_data_df)
    #     else
    #         data_x_full = data_x_fun(source_data_df, N)
    #     end
    # 
    #     data_y_full = Vector(source_data.ArrDelay .-15)
    #     data_x_full_frame = GLM.ModelFrame(model_formula, source_data, contrasts = Dict( :UniqueCarrier => GLM.DummyCoding(), :Origin => GLM.DummyCoding(), :Dest => GLM.DummyCoding()))
    #     x_names_full = coefnames(data_x_full_frame)
    # 
    #     omit_names = setdiff(x_names_full, x_names)
    #     omit_ind = []
    # 
    #     for name in omit_names
    #         tmp = findall(x->x==name, x_names_full)
    #         append!(omit_ind, tmp)
    #     end
    #     println("columns to drop ", omit_ind)
    #     trueparm = trueparm[setdiff(1:end, omit_ind)]
    # 
    #     data_x_full = data_x_full[1:end, setdiff(1:end, omit_ind)]
    #     parmdim = length(trueparm)
    # 
    #     println("The dimension of aligned true parm ", parmdim)
    #     println()
    #     if family == GLM.Poisson()
    #         if linkfun == GLM.LogLink()
    #             λ = exp.(data_x_full*trueparm)
    #             dλ_dlin = λ
    #             v = λ
    #             w = (dλ_dlin).^2 ./v
    #         elseif linkfun == GLM.CustomLink()
    #             η = data_x_full*trueparm
    #             λ = ifelse.(η .> 0, (1 .+ η .+ (η.^2)./2), exp.(η))
    #             dλ_dlin = ifelse.(η .> 0, 1 .+ η, exp.(η))
    #             v = λ
    #             w = (dλ_dlin).^2 ./v
    # 
    #             # η = data_x*trueparm
    #             # λ = log1pexp.(η)
    #             # dλ_dlin = exp.(η) ./ (one.(η) .+ exp.(η))
    #             # v = λ
    #             # w = (dλ_dlin).^2 ./v
    # 
    #             # η = data_x_full*trueparm
    #             # λ = ifelse.(η .> 0, 1 .+ η , exp.(η))
    #             # dλ_dlin = ifelse.(η .> 0, 1, exp.(η))
    #             # v = λ
    #             # w = (dλ_dlin).^2 ./v
    #         end
    #     elseif family == GLM.Binomial()
    #         if linkfun == GLM.LogitLink()
    #             logistic = x-> 1/(1+exp(-x))
    #             p = logistic.(data_x_full*trueparm)
    #             dp_dlin = (p .* (1 .-p))
    #             v = (p .* (1 .-p))
    #             w = (dp_dlin).^2 ./v
    #         elseif linkfun == GLM.ProbitLink()
    #             probit = x-> cdf(Normal(),x)
    #             lin = data_x_full*trueparm
    #             p = probit.(lin)
    #             dp_dlin = exp.(-lin.^2/2)/sqrt(2*pi)
    #             v = (p .* (1 .-p))
    #             w = (dp_dlin).^2 ./v
    #         elseif linkfun == GLM.CloglogLink()
    #             clogloginv = x-> 1-exp(-exp(x))
    #             p = clogloginv.(data_x_full*trueparm)
    #             dp_dlin = -(1 .-p).*log.(1 .-p)
    #             v = (p .* (1 .-p))
    #             w = (dp_dlin).^2 ./v
    #         end
    #     end
    #     w
    #     J = zeros(parmdim,parmdim)
    #     for i in 1:N
    #         J += data_x_full[i,:]*w[i]*data_x_full[i,:]'
    #     end
    #     J /= N
    # 
    #     Jinv = inv(J)
    #     sqrtJinv = sqrt(sqrt(Jinv*Jinv'))
    # 
    #     V = zeros(parmdim,parmdim)
    #     for i in 1:N
    #         V += data_x_full[i,:]*(data_y_full[i]-λ[i])^2*w[i]^2/dλ_dlin[i]^2*data_x_full[i,:]'
    #     end
    #     V /= N
    #     #@save trueparmfile parmdim Jinv V sqrtJinv trueparm model_formula data_x_fun linkfun
    # end


    # gradloglik = (parm,xy) -> xy[1]*(xy[2] - logit(xy[1]'*parm ))
    # gradlogprior = parm -> zeros(parmdim)

    # gradloglik = :poisson
    # gradlogprior = :flat
    gradloglik = :poisson
    #gradlogprior = :flat
    gradlogprior = :flat

    gradloglik_description = "(β,(x,y)) -> x*(y - exp(x'*β ))"
    gradlogprior_description = "β -> zeros(parmdim)"

    ismmap_init = false

    initparm = mleparm

    data_description = "Airline Dataset, Prepocessed by JN"
    println("data size ", size(data))

    @save fname parmdim Jinv V sqrtJinv data trueparm mleparm initparm gradloglik gradlogprior data_description samplesize ismmap_init
end

samplesize = 150000
fname = output_path_7*"airline-data-full-init-samplesize-"*string(samplesize)*post_fix1*".jld2"
source_data_df_file = output_path_7*"airline-data-full-df"*post_fix*".jld2"
trueparmfile = output_path_7*"airline-data-full-trueparm"*post_fix1*".jld2"
modelfile = output_path_7*"airline-data-full-model"*post_fix1*".jld2"


airlineInit_full(samplesize, fname, source_data_df_file, trueparmfile, modelfile)
