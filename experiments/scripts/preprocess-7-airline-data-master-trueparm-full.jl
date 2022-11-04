using JLD2
using StatsBase
using Statistics
using DataFrames
using RData
using Distributions
using Plots
using LinearAlgebra
using LogExpFunctions
using GLM

#include("preprocess-7-airline-data-full.jl")
output_path = "../output/"
isdir(output_path) || mkdir(output_path)
source_data_name = "airline-data-full-df"
include_str = true
post_fix = ""
post_fix1 = "-loglink-poisson"
if include_str
    post_fix *= "-all"
    #post_fix1 = "-uc-og-dt-custsqlink-binom"
    #post_fix1 = "-uc-custsqlink"
    #post_fix1 = "-uc-loglink"
    # post_fix1 = "-uc-logitlink-binom-whiteningSigmaX"
end


source_data_file = output_path*"experiments-7-airline-data-full/"*source_data_name*post_fix*".jld2"

output_path_7 = output_path*"experiments-7-airline-data-full/"
isdir(output_path_7) || mkdir(output_path_7)
fname = output_path_7*"airline-data-full-trueparm"*post_fix1*".jld2"
fname_model = output_path_7*"airline-data-full-model"*post_fix1*".jld2"
source_data = source_data_file
modelformula = @GLM.formula((ArrDelay - 15)~sin(Month/12)+cos(Month/12) + sin(DayOfWeek/7) + cos(DayOfWeek/7) + Distance + UniqueCarrier )
data_x_expr = ((mf, sd) -> GLM.ModelMatrix(GLM.ModelFrame(mf, sd, contrasts = Dict( :UniqueCarrier => GLM.DummyCoding()))).m)
linkfun=GLM.LogLink()

@eval function airlineTrueparm(fname, fname_model, source_data_file, modelformula, data_x_expr; linkfun=GLM.LogLink(), standardization = false)
    # source_data = (load(source_data_file)["airline"])[1:1000000,:]
    @load source_data_file source_data_df include_str
    
    print(size(source_data_df))
    # source_data = (load(source_data_file)["airline"])
    # source_data[:,"V5"] .*= 10

    N = (size(source_data_df)[1])
    println("size of data frame ",size(source_data_df))

    #add data whitening
    if include_str
        data_x = data_x_expr(modelformula, source_data_df)
    else
        data_x = data_x_expr(source_data_df, N)
    end
    name = ""

    data_y = Vector(source_data_df.ArrDelay .-15)
    
    family = GLM.Poisson()
    linkfun = GLM.LogLink()
    # family = GLM.Binomial()
    #Using whitening data for
    # d = length(names(dfWhitening))
    trueparm = GLM.coef(GLM.glm(modelformula, source_data_df, family, linkfun))
    println("True parameters is ", trueparm)
    # x_names = coefnames(data_x_frame)
    # println("The covariate with largest eigenvalue is ", x_names[ind])
    parmdim = length(trueparm)
    println("the dimension of true parm ", parmdim)
    GC.gc()

    if family == GLM.Poisson()
        if linkfun == GLM.LogLink()
            λ = exp.(data_x*trueparm)
            dλ_dlin = λ
            v = λ
            w = (dλ_dlin).^2 ./v
        end
    elseif family == GLM.Binomial()
        g(μ) = log(μ/(1-μ))
        dg(μ) = 1/(μ*(1-μ))
        d2g(μ) = -(1-2*μ)/(μ*(1-μ))^2
        if linkfun == GLM.LogitLink()
            h(η) = exp(η)/(one(η)+exp(η))
            dh(η) = exp(η)/(one(η)+exp(η))^2
            d2h(η) = (exp(η)*(one(η)-exp(η)))/(one(η)+exp(η))^3

            dloglik_dη(y,η) = (y-h(η))*dg(h(η))*dh(η)
            d2loglik_dη2(y,η) = -dg(h(η))*dh(η)^2 +  (y-h(η))*(d2g(h(η))*dh(η)^2 + dg(h(η))*d2h(η))
            dloglik_dβ(x,y,β) = dloglik_dη(y,x'*β)*x
            d2loglik_dβ2(x,y,β) = d2loglik_dη2(y,x'*β)*x*x'
        elseif linkfun == GLM.ProbitLink()
            probit = x-> cdf(Normal(),x)
            lin = data_x*trueparm
            p = probit.(lin)
            dp_dlin = exp.(-lin.^2/2)/sqrt(2*pi)
            v = (p .* (1 .-p))
            w = (dp_dlin).^2 ./v
        elseif linkfun == GLM.CloglogLink()
            clogloginv = x-> 1-exp(-exp(x))
            p = clogloginv.(data_x*trueparm)
            dp_dlin = -(1 .-p).*log.(1 .-p)
            v = (p .* (1 .-p))
            w = (dp_dlin).^2 ./v
        elseif linkfun == GLM.CustomLink()
            η = data_x*trueparm
            λ = ifelse.(η .> 0, (1 .+ η .+ (η.^2)./2), exp.(η))
            dλ_dlin = ifelse.(η .> 0, 1 .+ η, exp.(η))
            v = λ
            w = (dλ_dlin).^2 ./v
        end
    end
    if linkfun == GLM.LogLink()
        J = zeros(parmdim,parmdim)
        for i in 1:N
            J += data_x[i,:]*w[i]*data_x[i,:]'
        end
        J /= N

        Jinv = inv(J)
        sqrtJinv = sqrt(sqrt(Jinv*Jinv'))

        V = zeros(parmdim,parmdim)
        for i in 1:N
            V += data_x[i,:]*(data_y[i]-λ[i])^2*w[i]^2/dλ_dlin[i]^2*data_x[i,:]'
        end
        V /= N
    elseif linkfun == GLM.LogitLink()
        J = zeros(parmdim,parmdim)
        for i in 1:1
            # J += data_x[i,:]*p[i]*(1-p[i])*data_x[i,:]'
            J += -d2loglik_dβ2(data_x[i,:],data_y[i],trueparm)
        end
        J /= N
        println("J is ", J)
        Jinv = inv(J)
        println("J inverse is ", Jinv)
        sqrtJinv = sqrt(sqrt(Jinv*Jinv'))

        V = zeros(parmdim,parmdim)
        for i in 1:N
            # V += data_x[i,:]*(data_y[i]-p[i])^2 *data_x[i,:]'
            V += dloglik_dβ(data_x[i,:],data_y[i],trueparm)*dloglik_dβ(data_x[i,:],data_y[i],trueparm)'
        end
        V /= N
        println("V is ", V)
    end
    data_x_fun = data_x_expr
    @save fname parmdim Jinv V sqrtJinv trueparm name
    @save fname_model model_formula data_x_fun linkfun family
    N
end

model_formula = @GLM.formula((ArrDelay-15)~sin(Month/12)+cos(Month/12) + sin(DayOfWeek/7) + cos(DayOfWeek/7) + Distance + UniqueCarrier )


airlineTrueparm(output_path_7*"airline-data-full-trueparm"*post_fix1*".jld2", output_path_7*"airline-data-full-model"*post_fix1*".jld2",source_data_file,
            model_formula, data_x_expr; linkfun=linkfun, standardization = false)
