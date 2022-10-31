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

@eval function airlineTrueparmNB(fname, fname_model, source_data_file, parmdim, modelformula, data_x_expr; linkfun=LogLink())
    # source_data = (load(source_data_file)["airline"])[1:1000000,:]
    @load source_data_file source_data_df
    # source_data = (load(source_data_file)["airline"])
    # source_data[:,"V5"] .*= 10

    N = (size(source_data_df)[1])

    
    shuffle_indices=sample(1:N,N; replace=false, ordered=false)
    source_data = source_data_df[shuffle_indices,:]
    shuffle_indices=nothing
    source_data_df=nothing
    GC.gc()
    
    data_x = data_x_fun(source_data, N)
    data_y = Vector(source_data.ArrDelay .-15)
    
    trueparm = coef(glm(model_formula, source_data, Poisson(), linkfun))

    GC.gc()
    
    family = Poisson()
    # if family == Poisson()
    #     if linkfun == LogLink()
    #         λ = exp.(data_x*trueparm) 
    #         dλ_dlin = λ
    #         v = λ
    #         w = (dλ_dlin).^2 ./v
    #     end
    #     if linkfun == Special()
    #         λ = log.(1.0 .+ exp.(data_x*trueparm) )
    #         dλ_dlin = exp.(data_x*trueparm)./ exp.(λ)
    #         v = λ
    #         w = (dλ_dlin).^2 ./v
    #     end
    # elseif family == Binomial()
    #     if linkfun == LogitLink()
    #         logistic = x-> 1/(1+exp(-x))
    #         p = logistic.(data_x*trueparm) 
    #         dp_dlin = (p .* (1 .-p))
    #         v = (p .* (1 .-p))
    #         w = (dp_dlin).^2 ./v
    #     elseif linkfun == ProbitLink()           
    #         probit = x-> cdf(Normal(),x)
    #         lin = data_x*trueparm
    #         p = probit.(lin) 
    #         dp_dlin = exp.(-lin.^2/2)/sqrt(2*pi)
    #         v = (p .* (1 .-p))
    #         w = (dp_dlin).^2 ./v
    #     elseif linkfun == CloglogLink()
    #         clogloginv = x-> 1-exp(-exp(x))
    #         p = clogloginv.(data_x*trueparm) 
    #         dp_dlin = -(1 .-p).*log.(1 .-p)
    #         v = (p .* (1 .-p))
    #         w = (dp_dlin).^2 ./v
    #     end
    # end
    

    
    # g = canonical link. e.g. logit for bernoulli, and log for poisson
    # h = inverse link used, e.g. the inverse function of probit if you you want to do probit regression
    # dKf is the Kth derivative of f
    # g only depends on the glm model family (the likelihood for the response given the mean)
    # h only depends on the link function chosen
    if family == Poisson()
        g(μ) = log(μ)
        dg(μ) = 1/μ
        d2g(μ) = -1/μ^2
        if linkfun == LogLink()
            h(η) = exp(η)
            dh(η) = exp(η)
            d2h(η) = exp(η)
        end
        if linkfun == Special()
            h(η) = log1pexp(η)
            dh(η) = exp(η)/(one(η)+exp(η))
            d2h(η) = exp(η)/(one(η)+exp(η))^2
        end
    elseif family == Binomial()
        g(μ) = log(μ/(1-μ))
        dg(μ) = 1/(μ*(1-μ)) 
        d2g(μ) = -(1-2*μ)/(μ*(1-μ))^2
        if linkfun == LogitLink()
            h(η) = exp(η)/(one(η)+exp(η))
            dh(η) = exp(η)/(one(η)+exp(η))^2
            d2h(η) = (exp(η)*(one(η)-exp(η)))/(one(η)+exp(η))^3
        end
    end
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
    
    
    @save fname parmdim Jinv V sqrtJinv trueparm model_formula data_x_fun linkfun
    @save fname_model model_formula data_x_fun linkfun
    N
end

# Jinv smaller than sandwich, nointercept
    parmdim = 6
    model_formula = @formula((ArrDelay-15)~sin(Month/12)+cos(Month/12) + sin(DayOfWeek/7) + cos(DayOfWeek/7) + Distance)
    data_x_fun = ((source_data, N) -> [ones(N) sin.(source_data.Month/12) cos.(source_data.Month/12) sin.(source_data.DayOfWeek/7) cos.(source_data.DayOfWeek/7)  source_data.Distance])
    linkfun = LogLink()
    xidx=1
    yidx=3
    0.028128248076448627

    # parmdim = 4
    # model_formula = @formula((ArrDelay-15)~sin(Month/12) + sin(DayOfWeek/7) + Distance)
    # data_x_fun = ((source_data, N) -> [ones(N) sin.(source_data.Month/12) sin.(source_data.DayOfWeek/7)  source_data.Distance])
    # linkfun = LogLink()
    # xidx=1
    # yidx=3
    # 0.035488896955194416

    # parmdim = 7
    # model_formula = @formula((ArrDelay-15)~sin(Month/12)+cos(Month/12) + sin(DayOfWeek/7) + cos(DayOfWeek/7) + Distance + (DepTime ÷ 100))
    # data_x_fun = ((source_data, N) -> [ones(N) sin.(source_data.Month/12) cos.(source_data.Month/12) sin.(source_data.DayOfWeek/7) cos.(source_data.DayOfWeek/7)  source_data.Distance (source_data.DepTime .÷ 100)])
    # linkfun = LogLink()
    # xidx=6
    # yidx=7
    # 0.03313158031498715

    # parmdim = 6
    # model_formula = @formula((ArrDelay-15)~Month + DayOfWeek + Distance + (DepTime ÷ 100) + (DepTime % 100))
    # data_x_fun = ((source_data, N) -> [ones(N) source_data.Month source_data.DayOfWeek source_data.Distance (source_data.DepTime .÷ 100) (source_data.DepTime .% 100)])
    # linkfun = LogLink()
    # xidx=1
    # yidx=3
    # 0.099008299728362
    # 
    # parmdim = 4
    # model_formula = @formula((ArrDelay-15)~Month + DayOfWeek+(DepTime % 100))
    # data_x_fun = ((source_data, N) -> [ones(N) source_data.Month source_data.DayOfWeek (source_data.DepTime .% 100)])
    # linkfun = LogLink()
    # xidx=1
    # yidx=3
    # 0.017742288894497152

    # parmdim = 11
    # model_formula = @formula((ArrDelay-15)~sin(Month/12)+cos(Month/12)+sin(Month/6)+cos(Month/6) + sin(DayOfWeek/7) + cos(DayOfWeek/7) + Distance + (DepTime ÷ 100) + sin((DepTime % 100)/60) + cos((DepTime % 100)/60))
    # data_x_fun = ((source_data, N) -> [ones(N) sin.(source_data.Month/12) cos.(source_data.Month/12) sin.(source_data.Month/6) cos.(source_data.Month/6) sin.(source_data.DayOfWeek/7) cos.(source_data.DayOfWeek/7)  source_data.Distance (source_data.DepTime .÷ 100)  sin.((source_data.DepTime .% 100)/60) cos.((source_data.DepTime .% 100) ./ 60)])
    # linkfun = LogLink()
    # xidx=6
    # yidx=7
    # 0.04209326634548032
    
    # parmdim = 4
    # model_formula = @formula((ArrDelay-15)~ Month^2 + DayOfWeek^3 + (DepTime % 100))
    # data_x_fun = ((source_data, N) -> [ones(N) source_data.Month .^2 source_data.DayOfWeek.^3 (source_data.DepTime .% 100)])
    # linkfun = LogLink()
    # xidx=1
    # yidx=3
    # 0.004389146956800329
    
    # parmdim = 8
    # model_formula = @formula((ArrDelay-15)~Month + DayOfWeek + Distance + (DepTime ÷ 100) + Month^2 +Month^3 + (DepTime % 100))
    # data_x_fun = ((source_data, N) -> [ones(N) source_data.Month source_data.DayOfWeek source_data.Distance (source_data.DepTime .÷ 100) source_data.Month.^2 source_data.Month.^3  (source_data.DepTime .% 100)])
    # linkfun = LogLink()
    # xidx=1
    # yidx=5
    # 0.12293211632203127

airlineTrueparmNB(output_path_6*"airline-data-full-trueparm.jld2", output_path_6*"airline-data-full-model.jld2",source_data_file,
    parmdim, model_formula, data_x_fun; linkfun=linkfun)

@load output_path_6*"airline-data-full-trueparm.jld2"  parmdim Jinv V sqrtJinv trueparm

V

J = inv(Jinv)
J

sandwich = Jinv*V*Jinv

sqrt(sum((V-J).^2) / sum((V+J).^2/4))

C_Jinv = cov2cor(Jinv, sqrt.(diag(Jinv)))
C_sandwich = cov2cor(sandwich, sqrt.(diag(sandwich)))
maximum(abs.(C_sandwich - C_Jinv))

xrange = 3
xstep = xrange/100
X = (-xrange:xstep:xrange) * sqrt(max(Jinv[xidx,xidx],sandwich[xidx,xidx]))

yrange = 3
ystep = yrange/100
Y = (-yrange:ystep:yrange) * sqrt(max(Jinv[yidx,yidx],sandwich[yidx,yidx]))

Jinv = (Jinv+Jinv' )/2
sandwich = (sandwich+sandwich')/2
Z_Jinv = [pdf(MvNormal([0,0], Jinv[[xidx,yidx],[xidx,yidx]]), [x,y]) for x in X, y in Y]
Z_sandwich = [pdf(MvNormal([0,0], sandwich[[xidx,yidx],[xidx,yidx]]), [x,y]) for x in X, y in Y]

contour(X,Y,Z_Jinv, linewidth=2 , colorbar = false, seriescolor = cgrad(:reds), levels = 5)
contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_sandwich, linewidth=2 , colorbar = false, seriescolor=cgrad(:blues), levels = 5)



# 
# 
# 
using LogExpFunctions
####
struct customLink <: Link01 end
"""
    GLM.linkfun(L::Link, μ::Real)
Return `η`, the value of the linear predictor for link `L` at mean `μ`.
"""

linkfun(::customLink, μ::Real) = logexpm1(μ)
linkinv(::customLink, η::Real) = log1pexp(η)
mueta(::customLink, η::Real) = exp(η) / (one(η)+ exp(η))
function inverselink(::customLink, η::Real)
    return logexpm1(μ), exp(η) / (one(η)+ exp(η)), convert(float(typeof(η)), NaN)
end
