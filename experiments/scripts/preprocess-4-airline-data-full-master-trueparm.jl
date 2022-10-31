using JLD2
using StatsBase
using Statistics
using GLM
using DataFrames
using RData

output_path = "../output/"
isdir(output_path) || mkdir(output_path)

source_data_file = "../source-data/data_airline_raw.rdata"
## experiment-3 Logistic REgression Experiments

output_path_4 = output_path*"experiments-4-airline-data/"
isdir(output_path_4) || mkdir(output_path_4)

# output_path_5 = output_path*"experiments-5-airline-data-full/"
# isdir(output_path_5) || mkdir(output_path_5)

function airlineTrueparm(fname, source_data_file)
    source_data = load(source_data_file)["airline"]

    N = (size(source_data)[1])
    parmdim = 4
    
    shuffle_indices=sample(1:N,N; replace=false, ordered=false)
    source_data = source_data[shuffle_indices,:]
    shuffle_indices=nothing
    GC.gc()
    
    data_x = [ones(N) Matrix(source_data[:,["V3", "V4", "V5"]])]
    data_y = Vector(source_data[:,"V2"])

    trueparm = coef(glm(@formula(V2~V3+V4+V5), source_data, Binomial()))
    GC.gc()
    
    logit = x-> 1/(1+exp(-x))
    p = logit.(data_x*trueparm) 
    
    
    g(μ) = log(μ/(1-μ))
    dg(μ) = 1/(μ*(1-μ)) 
    d2g(μ) = -(1-2*μ)/(μ*(1-μ))^2
    h(η) = exp(η)/(one(η)+exp(η))
    dh(η) = exp(η)/(one(η)+exp(η))^2
    d2h(η) = (exp(η)*(one(η)-exp(η)))/(one(η)+exp(η))^3
    
    dloglik_dη(y,η) = (y-h(η))*dg(h(η))*dh(η) 
    d2loglik_dη2(y,η) = -dg(h(η))*dh(η)^2 +  (y-h(η))*(d2g(h(η))*dh(η)^2 + dg(h(η))*d2h(η))
    
    dloglik_dβ(x,y,β) = dloglik_dη(y,x'*β)*x
    d2loglik_dβ2(x,y,β) = d2loglik_dη2(y,x'*β)*x*x'
    
    J = zeros(parmdim,parmdim) 
    for i in 1:N
        # J += data_x[i,:]*p[i]*(1-p[i])*data_x[i,:]'
        J += -d2loglik_dβ2(data_x[i,:],data_y[i],trueparm)
    end
    J /= N

    Jinv = inv(J)
    sqrtJinv = sqrt(sqrt(Jinv*Jinv'))

    V = zeros(parmdim,parmdim)  
    for i in 1:N
        # V += data_x[i,:]*(data_y[i]-p[i])^2 *data_x[i,:]'
        V += dloglik_dβ(data_x[i,:],data_y[i],trueparm)*dloglik_dβ(data_x[i,:],data_y[i],trueparm)'
    end
    V /= N
    
    @save fname*".jld2" parmdim Jinv V sqrtJinv trueparm
end

# @save output_path_4*"airline-data-trueparm.jld2" parmdim Jinv V sqrtJinv trueparm

airlineTrueparm(output_path_4*"airline-data-trueparm",source_data_file)

# airlineTrueparm(output_path_5*"airline-data-full-trueparm",source_data_file)