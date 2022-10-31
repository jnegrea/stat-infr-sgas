## Generic Run and Plot Utils for modelInit objects

function modelRun(modelinit, fname, algos, samplesize, batchsizes, w, numstepss; multithread=true, ismmap_run=false, keepperiods=false, fnameprefix=fname)
    @load modelinit parmdim Jinv V sqrtJinv data trueparm mleparm initparm gradloglik gradlogprior data_description samplesize ismmap_init

    Vinv = inv(sqrt(V*V'))

    if gradloglik == :logistic
        logit = x-> 1/(1+exp(-x))
        gradloglik = (parm,xy) -> xy[1]*(xy[2] - logit(xy[1]'*parm ))
    elseif gradloglik == :poisson
        gradloglik = (parm,xy) -> xy[1]*(xy[2] - exp(xy[1]'*parm ))
    end
    if gradlogprior == :flat
        gradlogprior = parm -> zeros(parmdim)
    end

    if ismmap_run==false
        ismmap_run=ismmap_init
    end

    if (keepperiods==false)
        keepperiods = Dict(algo => 1 for algo in algos)
    end

    if multithread
        parms = sgldMultiPreSpecMT(algos, initparm, data, gradloglik, gradlogprior, samplesize, batchsizes,
            parmdim, numstepss, Jinv, Vinv, sqrtJinv, w, ismmap_init, ismmap_run; keepperiods=keepperiods, fnameprefix=fnameprefix)
    else
        parms = sgldMultiPreSpec(algos, initparm, data, gradloglik, gradlogprior, samplesize, batchsizes,
            parmdim, numstepss, Jinv, Vinv, sqrtJinv, w, ismmap_init, ismmap_run; keepperiods=keepperiods, fnameprefix=fnameprefix)
    end


    if ( ismmap_init && (mleparm[1] == :run) )
        mleparm = mean([mean(parms[algo]) for algo in algos])
    elseif ( ismmap_init && (mleparm[1] == :glm) )
        mleparm = mleparm[2]
    end

    # @save fname*".jld2" algos parms trueparm mleparm Jinv V sqrtJinv w samplesize batchsizes numstepss ismmap_init ismmap_run keepperiods
    @save fname algos parms trueparm mleparm Jinv V sqrtJinv w samplesize batchsizes numstepss ismmap_init ismmap_run keepperiods
end

function modelPlot(modelrunfile, pltfname, layers, xlimvals, ylimvals; nameattr= false, levels=30, localscale=true, fontscale=1, legendloc=:topleft, iteravg_numepochs=1)
    @load modelrunfile algos parms trueparm mleparm Jinv V sqrtJinv w samplesize batchsizes numstepss ismmap_init ismmap_run keepperiods
    sgldplot(pltfname, samplesize, algos, parms, trueparm, mleparm, Jinv, sqrtJinv, V, w, layers, xlimvals, ylimvals, ismmap_run, numstepss, batchsizes, keepperiods; nameattr= false, levels=levels, localscale=localscale, fontscale=fontscale, legendloc=legendloc, iteravg_numepochs=iteravg_numepochs) 
end

function modelSummary(modelinit, modelrunfile, fname)
    @load modelinit parmdim Jinv V sqrtJinv data trueparm mleparm initparm gradloglik gradlogprior data_description samplesize ismmap_init
    @load modelrunfile algos parms trueparm mleparm Jinv V sqrtJinv w samplesize batchsizes numstepss ismmap_init ismmap_run keepperiods

    localmap = p -> sqrt(samplesize)*(p .- mleparm)

    if !ismmap_run
        local_parms = Dict(algo => localmap.(parm) for (algo,parm) in parms)
    else
        function reader(algo)
            io2 = open(parms[algo][1]*".bin")
            parms_values = Mmap.mmap(io2, parms[algo][2], numstepss[algo]÷keepperiods[algo])
            close(io2)
            local_parms = localmap.(parms_values)
        end
        local_parms = Dict(algo => reader(algo) for algo in algos)
    end

    local_trueparm = sqrt(samplesize)*(trueparm-mleparm)

    IACTalgos = Dict(algo => iact_clean(local_parms[algo], parmdim, samplesize;
        extravec1=real.(eigen(Jinv).vectors[:,parmdim]),
        extravec2=real.(eigen(Jinv*V).vectors[:,parmdim]))*keepperiods[algo]*batchsizes[algo] for algo in algos)
    IACTalgosstring = ""
    for algo in algos
        IACTalgosstring *= "IACT_"*string(algo)*" = "*string(IACTalgos[algo])*"\n"
    end

    IACTtheorystring=""
    if :vanillaSGD in algos
        IACT_esgd = round(maximum(real.(eigen(Jinv).values)),sigdigits=2)
        IACTtheorystring *= "IACT_esgd = "*string(IACT_esgd)*"\n"
    end
    if :preconSGD in algos
        IACT_sandwich = round(1,sigdigits=2)
        IACTtheorystring *= "IACT_precondSGD = "*string(IACT_sandwich)*"\n"
    end
    if :preconSGLD in algos
        IACT_mix = round(1/(1-w),sigdigits=2)
        IACTtheorystring *= "IACT_preconSGLD = "*string(IACT_mix)*"\n"
    end
    if :preconVSGD in algos
        IACT_vsgd = round(maximum(real.(eigen(Jinv*V).values)),sigdigits=2)
        IACTtheorystring *= "IACT_preconVSGD = "*string(IACT_vsgd)*"\n"
    end

    IACT_sharedstring =""
    for algo in algos
        IACT_sharedstring *= "IACT for "*string(algo)*":\n"
        IACT_sharedstring *= "    Empirical = "*string(IACTalgos[algo])*"\n"
        if algo==:vanillaSGD
            IACT_esgd = round(maximum(real.(eigen(Jinv).values)),sigdigits=2)
            IACT_sharedstring *= "    Expected = "*string(IACT_esgd)*"\n"
        end
        if algo==:preconSGD
            IACT_sandwich = round(1,sigdigits=2)
            IACT_sharedstring *= "    Expected = "*string(IACT_sandwich)*"\n"
        end
        if algo==:preconSGLD
            IACT_mix = round(1/(1-w),sigdigits=2)
            IACT_sharedstring *= "    Expected = "*string(IACT_mix)*"\n"
        end
        if algo==:preconVSGD
            IACT_vsgd = round(maximum(real.(eigen(Jinv*V).values)),sigdigits=2)
            IACT_sharedstring *= "    Expected = "*string(IACT_vsgd)*"\n"
        end
    end

    # if :largeBatchPreconSGLD in algos
    #     IACT_J = round(1/batchsizes[:largeBatchPreconSGLD],sigdigits=2)
    #     IACTtheorystring *= "IACT_J = "*string(IACT_J)*"\n"
    # end
    # if :largeBatchVanillaSGLD in algos
    #     IACT_J = round(1/(4*batchsizes[:largeBatchVanillaSGLD])*maximum(eigen(Jinv).values),sigdigits=2)
    #     IACTtheorystring *= "IACT_J = "*string(IACT_J)*"\n"
    # end
    #
    pval_esgd = round.(ccdf.(Chisq(parmdim), local_trueparm' *((sqrtJinv*V*sqrtJinv)\ local_trueparm)), sigdigits=2)
    pval_sandwich = round.(ccdf.(Chisq(parmdim), local_trueparm'*( (Jinv*V*Jinv)\ local_trueparm)), sigdigits=2)
    pval_Jinv = round.(ccdf.(Chisq(parmdim), local_trueparm'* ((Jinv)\ local_trueparm)), sigdigits=2)
    pval_mix = round.(ccdf.(Chisq(parmdim), local_trueparm'* ((w*Jinv+(1-w)*Jinv*V*Jinv)\ local_trueparm)), sigdigits=2)



    description = "samplesize = "*string(samplesize)*"\n"*
        "batchsizes = "*string(batchsizes)*"\n"*
        "parmdim = "*string(parmdim)*"\n"*
        "numstepss = "*string(numstepss)*"\n"*
        "w = "*string(w)*"\n"*
        "data = "*data_description*"\n"*
        "trueparm = "*string(trueparm)*"\n"*
        "mleparm = "*string(mleparm)*"\n"*
        "initparm = "*string(initparm)*"\n"*
        # IACTalgosstring*
        # IACTtheorystring*
        IACT_sharedstring*
        "pval_esgd = "*string(pval_esgd)*"\n"*
        "pval_sandwich = "*string(pval_sandwich)*"\n"*
        "pval_Jinv = "*string(pval_Jinv)*"\n"*
        "pval_mix = "*string(pval_mix)
    io=open(fname, "w+")
    write(io,description)
    close(io)
end


## Gaussian Location modelInit
function gaussianLocationInit(samplesize, parmdim, Σ, Σlik, fname; initparm=false)
    Jinv = Σlik
    V = inv(Σlik)*Σ*inv(Σlik)
    sqrtJinv = sqrt(sqrt(Jinv*Jinv'))

    data = [rand(MvNormal(zeros(parmdim), Σ),1) for i in 1:samplesize]
    trueparm = zeros(parmdim)
    mleparm = mean(data)
    if !initparm
        initparm = mleparm
    end

    gradloglik = (parm,x) -> - Σlik\(parm-x)
    gradlogprior = parm -> zeros(parmdim)

    data_description = "[rand(MvNormal(zeros(parmdim), Σ),samplesize)[:,i] for i in 1:samplesize]"
    gradloglik_description = "(parm,x) -> - Σlik\\(parm-x)"
    gradlogprior_description = "parm -> zeros(parmdim)"

    ismmap_init = false

    @save fname parmdim Jinv V sqrtJinv data trueparm mleparm initparm gradloglik gradlogprior data_description samplesize ismmap_init
end

## LinearRegression modelInit
function linearRegressionInit(samplesize, parmdim, Σcov, β,σ2, fname; initparm=false)

    Jinv = inv(sqrt(Σcov*Σcov'))
    V = σ2*Σcov
    sqrtJinv = sqrt(sqrt(Jinv*Jinv'))

    data_x = rand(MvNormal(zeros(parmdim), Σcov),samplesize)
    data_y = [rand(Normal(data_x[:,i]'*β,sqrt(σ2))) for i in 1:samplesize]
    data = [(data_x[:,i],data_y[i]) for i in 1:samplesize]
    trueparm = β
    mleparm = (data_x*data_x')\(data_x*data_y)

    if !initparm
        initparm = mleparm
    end

    gradloglik = (parm,xy) -> -xy[1]*(xy[1]'*parm - xy[2])
    gradlogprior = parm -> zeros(parmdim)

    data_description = "X~N(0,Σcov), Y|X~N(Xβ,σ2)"
    gradloglik_description = "(β,(x,y)) -> -x(x'β-y)"
    gradlogprior_description = "β -> zeros(parmdim)"

    ismmap_init = false
    @save fname parmdim Jinv V sqrtJinv data trueparm mleparm initparm gradloglik gradlogprior data_description samplesize ismmap_init
end

## LogisticRegression modelInit
function logisticRegressionInit(samplesize, parmdim, Σcov, β, fname; outof=:one, initparm=false, mc_size=Int(1e7))
    logit = x-> 1/(1+exp(-x))
    if outof == :one
        data_n = Int.(ones(samplesize))
        mc_data_n = Int.(ones(mc_size))
        data_description = "X~N(0,Σcov), Y|X,N~Binomial(1, logit(Xβ))"
    elseif outof[1] ==:poisson
        λ = outof[2]
        data_n = 1 .+Int.(rand(Poisson(λ),samplesize))
        mc_data_n = 1 .+Int.(rand(Poisson(λ),mc_size))
        data_description = "X~N(0,Σcov), N~Poisson(outof[2]), Y|X,N~Binomial(N, logit(Xβ))"
    end
    mc_data_x = rand(MvNormal(zeros(parmdim), Σcov),mc_size)
    mc_p = [logit(mc_data_x[:,i]'*β) for i in 1:mc_size]

    mc_data_y = [rand(Binomial(mc_data_n[i],mc_p[i])) for i in 1:mc_size]

    # J = mean([mc_data_n[i]*mc_data_x[:,i]*mc_p[i]*(1-mc_p[i])*mc_data_x[:,i]' for i in 1:mc_size])
    # Jinv = inv(J)
    #
    # V = mean([mc_data_x[:,i]*(mc_data_y[i]-mc_data_n[i]*mc_p[i])^2 *mc_data_x[:,i]' for i in 1:mc_size])
    g(μ) = log(μ/(1-μ))
    dg(μ) = 1/(μ*(1-μ))
    d2g(μ) = -(1-2*μ)/(μ*(1-μ))^2
    h(η) = exp(η)/(one(η)+exp(η))
    dh(η) = exp(η)/(one(η)+exp(η))^2
    d2h(η) = (exp(η)*(one(η)-exp(η)))/(one(η)+exp(η))^3

    dloglik_dη(y,n,η) = (y-n*h(η))*dg(h(η))*dh(η)
    d2loglik_dη2(y,n,η) = -n*dg(h(η))*dh(η)^2 +  (y-n*h(η))*(d2g(h(η))*dh(η)^2 + dg(h(η))*d2h(η))

    dloglik_dβ(x,y,n,β1) = dloglik_dη(y,n,x'*β1)*x
    d2loglik_dβ2(x,y,n,β1) = d2loglik_dη2(y,n,x'*β1)*x*x'

    J = zeros(parmdim,parmdim)
    for i in 1:mc_size
        # J += data_x[i,:]*p[i]*(1-p[i])*data_x[i,:]'
        J += -d2loglik_dβ2(mc_data_x[:,i],mc_data_y[i],mc_data_n[i],β)
    end
    J /= mc_size

    Jinv = inv(J)
    sqrtJinv = sqrt(sqrt(Jinv*Jinv'))

    V = zeros(parmdim,parmdim)
    for i in 1:mc_size
        # V += data_x[i,:]*(data_y[i]-p[i])^2 *data_x[i,:]'
        V += dloglik_dβ(mc_data_x[:,i],mc_data_y[i],mc_data_n[i],β)*dloglik_dβ(mc_data_x[:,i],mc_data_y[i],mc_data_n[i],β)'
    end
    V /= mc_size

    data_x = rand(MvNormal(zeros(parmdim), Σcov),samplesize)
    p = [logit(data_x[:,i]'*β) for i in 1:samplesize]
    data_y = [rand(Binomial(data_n[i],p[i])) for i in 1:samplesize]
    data = [(data_x[:,i],data_y[i],data_n[i]) for i in 1:samplesize]
    trueparm = β

    model_data = DataFrame(Tables.table(data_x', header = Symbol.(:x, axes(data_x', 2))))
    # model_data.Y = data_y
    model_data.Yprop = data_y./data_n
    model_formula = term(:Yprop)~term(0)+sum(term.(Symbol.(names(model_data[:,Not(:Yprop)]))))
    mleparm = coef(glm(model_formula,model_data,Binomial(),wts = data_n))

    if !initparm
        initparm = mleparm
    end

    gradloglik = (parm,xyn) -> xyn[1]*(xyn[2] - xyn[3]*logit(xyn[1]'*parm ))
    gradlogprior = parm -> zeros(parmdim)

    gradloglik_description = "(β,(x,y,n)) -> x*(y - n*logit(x'*β ))"
    gradlogprior_description = "β -> zeros(parmdim)"

    ismmap_init = false
    @save fname parmdim Jinv V sqrtJinv data trueparm mleparm initparm gradloglik gradlogprior data_description samplesize ismmap_init
end


## LogisticRegression modelInit for very big samplesizes

# mleparmtype options:
#   :run    estimate it from the runs in a post-hoc way (may be imprecise)
#   :glm    compute it with the glm package (may be slow)

function logisticRegressionInitHuge(samplesize, parmdim, Σcov, β, fname; outof=:one, initparm=false, mc_size=Int(1e7), atatime=10000, mleparmtype=:glm)
    logit = x-> 1/(1+exp(-x))
    print("start monte carlo\n")

    if outof == :one
        mc_data_n = Int.(ones(mc_size))
        data_description = "X~N(0,Σcov), Y|X,N~Binomial(1, logit(Xβ))"
    elseif outof[1] ==:poisson
        λ = outof[2]
        mc_data_n = Int.(rand(Poisson(λ),mc_size))
        data_description = "X~N(0,Σcov), N~Poisson(outof[2]), Y|X,N~Binomial(N, logit(Xβ))"
    end

    mc_data_x = rand(MvNormal(zeros(parmdim), Σcov),mc_size)
    mc_p = [logit(mc_data_x[:,i]'*β) for i in 1:mc_size]

    mc_data_y = [rand(Binomial(mc_data_n[i],mc_p[i])) for i in 1:mc_size]

    # J = mean([mc_data_n[i]*mc_data_x[:,i]*mc_p[i]*(1-mc_p[i])*mc_data_x[:,i]' for i in 1:mc_size])
    # Jinv = inv(J)
    #
    # V = mean([mc_data_x[:,i]*(mc_data_y[i]-mc_data_n[i]*mc_p[i])^2 *mc_data_x[:,i]' for i in 1:mc_size])
    #
    # sqrtJinv = sqrt(sqrt(Jinv*Jinv'))
    g(μ) = log(μ/(1-μ))
    dg(μ) = 1/(μ*(1-μ))
    d2g(μ) = -(1-2*μ)/(μ*(1-μ))^2
    h(η) = exp(η)/(one(η)+exp(η))
    dh(η) = exp(η)/(one(η)+exp(η))^2
    d2h(η) = (exp(η)*(one(η)-exp(η)))/(one(η)+exp(η))^3

    dloglik_dη(y,n,η) = (y-n*h(η))*dg(h(η))*dh(η)
    d2loglik_dη2(y,n,η) = -n*dg(h(η))*dh(η)^2 +  (y-n*h(η))*(d2g(h(η))*dh(η)^2 + dg(h(η))*d2h(η))

    dloglik_dβ(x,y,n,β1) = dloglik_dη(y,n,x'*β1)*x
    d2loglik_dβ2(x,y,n,β1) = d2loglik_dη2(y,n,x'*β1)*x*x'

    J = zeros(parmdim,parmdim)
    for i in 1:mc_size
        # J += data_x[i,:]*p[i]*(1-p[i])*data_x[i,:]'
        J += -d2loglik_dβ2(mc_data_x[:,i],mc_data_y[i],mc_data_n[i],β)
    end
    J /= mc_size

    Jinv = inv(J)
    sqrtJinv = sqrt(sqrt(Jinv*Jinv'))

    V = zeros(parmdim,parmdim)
    for i in 1:mc_size
        # V += data_x[i,:]*(data_y[i]-p[i])^2 *data_x[i,:]'
        V += dloglik_dβ(mc_data_x[:,i],mc_data_y[i],mc_data_n[i],β)*dloglik_dβ(mc_data_x[:,i],mc_data_y[i],mc_data_n[i],β)'
    end
    V /= mc_size

    data1 = fname*"-data"
    data2 = Vector{Tuple{SVector{parmdim,Float64}, Int64, Int64}}
    data = (data1,data2)

    print("initialize mmap object\n")

    io = open(data1*".bin", "w+")
    dataset = Mmap.mmap(io, data2, samplesize)
    processed_so_far=0

    trueparm = β

    print("start looping\n")

    while processed_so_far < samplesize
        subsetsamplesize = min(atatime, samplesize - processed_so_far)
        if outof == :one
            data_n = Int.(ones(subsetsamplesize))
        elseif outof[1] ==:poisson
            λ = outof[2]
            data_n = Int.(rand(Poisson(λ),subsetsamplesize))
        end

        data_x = rand(MvNormal(zeros(parmdim), Σcov),subsetsamplesize)
        p = [logit(data_x[:,i]'*β) for i in 1:subsetsamplesize]
        data_y = [rand(Binomial(data_n[i],p[i])) for i in 1:subsetsamplesize]
        data_xyn = [(data_x[:,i],data_y[i],data_n[i]) for i in 1:subsetsamplesize]
        dataset[processed_so_far .+ (1:subsetsamplesize)] = data_xyn
        Mmap.sync!(dataset)
        processed_so_far+=subsetsamplesize
        print("processed so far = "*string(processed_so_far)*"\n")
    end
    close(io)

    if mleparmtype == :glm
        print("computing mle with glm\n")
        print("making dataframe \n")
        # model_data = DataFrame(Tables.table(vcat(Array((xyn->xyn[1]').(dataset))...), header = Symbol.(:x, 1:parmdim)))
        print("test: array \n")
        a =Array((xyn->Array(xyn[1])').(dataset))

        print("test: vcat \n")
        v = vcat(a...)

        print("test: table \n")
        t = Tables.table(v, header = Symbol.(:x, 1:parmdim))

        print("test: dataframe \n")
        model_data = DataFrame(t)


        # model_data.Y = data_y
        model_data.Yprop = (xyn->xyn[2]/xyn[3]).(dataset)
        model_formula = term(:Yprop)~term(0)+sum(term.(Symbol.(names(model_data[:,Not(:Yprop)]))))

        print("running glm \n")
        mleparm = (:glm, coef(glm(model_formula,model_data,Binomial(),wts = (xyn->xyn[3]).(dataset))))
    elseif mleparmtype == :run
        mleparm = (:run, nothing)
    else
        error("big OOF: the mleparmtype is not valid")
    end


    if !initparm
        if mleparmtype == :glm
            initparm = mleparm[2]
        else
            # initparm = zeros(parmdim)
            initparm = trueparm
        end
    end

    print("wrapping up logisticRegressionInitHuge\n")
    dataset=nothing

    gradloglik = (parm,xyn) -> xyn[1]*(xyn[2] - xyn[3]*logit(xyn[1]'*parm ))
    gradlogprior = parm -> zeros(parmdim)

    gradloglik_description = "(β,(x,y,n)) -> x*(y - n*logit(x'*β ))"
    gradlogprior_description = "β -> zeros(parmdim)"

    fname2 = fname*".jld2"
    ismmap_init = true
    @save fname2 parmdim Jinv V sqrtJinv data trueparm mleparm initparm gradloglik gradlogprior data_description samplesize ismmap_init
end


function airlineInit(samplesize, fname, source_data_file; trueparmfile = false, initparm=false,  mc_size=false, mleparmtype=:glm)
    source_data = load(source_data_file)["airline"]

    N = (size(source_data)[1])
    parmdim = 4

    if trueparmfile == false
        if mc_size==false
            mc_size = N
        end

        mc_indexes = sample(1:N, mc_size; replace=false, ordered=false)
        mc_data = source_data[mc_indexes,:]
        mc_data_x = [ones(mc_size) Matrix(mc_data[:,["V3", "V4", "V5"]])]
        mc_data_y = Vector(mc_data[:,"V2"])

        trueparm = coef(glm(@formula(V2~V3+V4+V5), mc_data, Binomial()))
        GC.gc()

        mc_p = logit.(mc_data_x*trueparm)

        # J = zeros(parmdim,parmdim)
        # for i in 1:mc_size
        #     J += mc_data_x[:,i]*mc_p[i]*(1-mc_p[i])*mc_data_x[:,i]'
        # end
        # J /= mc_size
        #
        # Jinv = inv(J)
        # sqrtJinv = sqrt(sqrt(Jinv*Jinv'))
        #
        # V = zeros(parmdim,parmdim)
        # for i in 1:mc_size
        #     V += mc_data_x[:,i]*(mc_data_y[i]-mc_p[i])^2 *mc_data_x[:,i]'
        # end
        # V /= mc_size
        g(μ) = log(μ/(1-μ))
        dg(μ) = 1/(μ*(1-μ))
        d2g(μ) = -(1-2*μ)/(μ*(1-μ))^2
        h(η) = exp(η)/(one(η)+exp(η))
        dh(η) = exp(η)/(one(η)+exp(η))^2
        d2h(η) = (exp(η)*(one(η)-exp(η)))/(one(η)+exp(η))^3

        dloglik_dη(y,η) = (y-h(η))*dg(h(η))*dh(η)
        d2loglik_dη2(y,η) = -dg(h(η))*dh(η)^2 +  (y-h(η))*(d2g(h(η))*dh(η)^2 + dg(h(η))*d2h(η))

        dloglik_dβ(x,y,β1) = dloglik_dη(y,n,x'*β1)*x
        d2loglik_dβ2(x,y,β1) = d2loglik_dη2(y,x'*β1)*x*x'

        J = zeros(parmdim,parmdim)
        for i in 1:mc_size
            # J += data_x[i,:]*p[i]*(1-p[i])*data_x[i,:]'
            J += -d2loglik_dβ2(mc_data_x[:,i],mc_data_y[i],β)
        end
        J /= mc_size

        Jinv = inv(J)
        sqrtJinv = sqrt(sqrt(Jinv*Jinv'))

        V = zeros(parmdim,parmdim)
        for i in 1:mc_size
            # V += data_x[i,:]*(data_y[i]-p[i])^2 *data_x[i,:]'
            V += dloglik_dβ(mc_data_x[:,i],mc_data_y[i],β)*dloglik_dβ(mc_data_x[:,i],mc_data_y[i],β)'
        end
        V /= mc_size
    else
        @load trueparmfile parmdim Jinv V sqrtJinv trueparm
    end

    sample_indexes = sample(1:N, samplesize; replace=false, ordered=false)
    sample_data = source_data[sample_indexes,:]
    # sample_data_x = [ones(samplesize) Matrix(sample_data[:,["V3", "V4", "V5"]])]
    # sample_data_y = Vector(sample_data[:,"V2"])

    source_data = nothing
    GC.gc()

    mleparm = coef(glm(@formula(V2~V3+V4+V5), sample_data, Binomial()))

    data = [([1;Vector(sample_data[i,["V3", "V4", "V5"]])],sample_data[i,"V2"]) for i in 1:samplesize]


    # gradloglik = (parm,xy) -> xy[1]*(xy[2] - logit(xy[1]'*parm ))
    # gradlogprior = parm -> zeros(parmdim)

    gradloglik = :logistic
    gradlogprior = :flat

    gradloglik_description = "(β,(x,y)) -> x*(y - logit(x'*β ))"
    gradlogprior_description = "β -> zeros(parmdim)"

    ismmap_init = false

    if !initparm
        if mleparmtype == :glm
            initparm = mleparm
        else
            initparm = trueparm
        end
    end

    data_description = "Airline Dataset, Prepocessed by Pollock et al. (2020)"

    @save fname*".jld2" parmdim Jinv V sqrtJinv data trueparm mleparm initparm gradloglik gradlogprior data_description samplesize ismmap_init
end


#### NOT UPDATED WITH THE FIXED GLM SECOND MOMENT!!!
function airlineInit_altmodel(samplesize, fname, source_data_file, model_formula, data_x_fun; trueparmfile = false, initparm=false,  mc_size=false, mleparmtype=:glm, linkfun=LogitLink())
    print("NOT UPDATED WITH THE FIXED GLM SECOND MOMENT!!!")
    source_data = load(source_data_file)["airline"]

    N = (size(source_data)[1])
    parmdim = 4

    if trueparmfile == false
        if mc_size==false then
            mc_size = N
        end

        mc_indexes = sample(1:N, mc_size; replace=false, ordered=false)
        mc_data = source_data[mc_indexes,:]
        mc_data_x = data_x_fun(mc_data, mc_size)
        # mc_data_x = [ones(mc_size) Matrix(mc_data[:,["V3", "V4", "V5"]])]
        mc_data_y = Vector(mc_data[:,"V2"])

        trueparm = coef(glm(model_formula, mc_data, Binomial(), linkfun))
        GC.gc()

        # mc_p = logit.(mc_data_x*trueparm)
        #
        # J = zeros(parmdim,parmdim)
        # for i in 1:mc_size
        #     J += mc_data_x[:,i]*mc_p[i]*(1-mc_p[i])*mc_data_x[:,i]'
        # end
        # J /= mc_size
        #
        # Jinv = inv(J)
        # sqrtJinv = sqrt(sqrt(Jinv*Jinv'))
        #
        # V = zeros(parmdim,parmdim)
        # for i in 1:mc_size
        #     V += mc_data_x[:,i]*(mc_data_y[i]-mc_p[i])^2 *mc_data_x[:,i]'
        # end
        # V /= mc_size

        if linkfun == LogitLink()
            logistic = x-> 1/(1+exp(-x))
            mc_p = logistic.(mc_data_x*trueparm)
            mc_dp_dlin = 1 ./(mc_p .* (1 .-mc_p))
            mc_v = (mc_p .* (1 .-mc_p))
            mc_w = (mc_dp_dlin).^2 ./mc_v
        elseif linkfun == ProbitLink()
            probit = x-> cdf(Normal(),x)
            mc_p = probit.(mc_data_x*trueparm)
            dp_dlin = exp.(-p.^2/2)/sqrt(2*pi)
            mc_v = (mc_p .* (1 .-mc_p))
            mc_w = (mc_dp_dlin).^2 ./mc_v
        elseif linkfun == CloglogLink()
            clogloginv = x-> 1-exp(-exp(x))
            mc_p = clogloginv.(mc_data_x*trueparm)
            mc_dp_dlin = -(1 .-mc_p).*log.(1 .-mc_p)
            mc_v = (mc_p .* (1 .-mc_p))
            mc_w = (mc_dp_dlin).^2 ./mc_v
        end

        J = zeros(parmdim,parmdim)
        for i in 1:mc_size
            J += mc_data_x[:,i]*mc_w[i]*mc_data_x[:,i]'
        end
        J /= mc_size

        Jinv = inv(J)
        sqrtJinv = sqrt(sqrt(Jinv*Jinv'))

        V = zeros(parmdim,parmdim)
        for i in 1:mc_size
            V += mc_data_x[:,i]*(mc_data_y[i]-mc_p[i])^2*mc_w[i]^2/mc_dp_dlin[i]^2*mc_data_x[:,i]'
        end
        V /= mc_size
    else
        @load trueparmfile parmdim Jinv V sqrtJinv trueparm
    end

    sample_indexes = sample(1:N, samplesize; replace=false, ordered=false)
    sample_data = source_data[sample_indexes,:]
    # sample_data_x = [ones(samplesize) Matrix(sample_data[:,["V3", "V4", "V5"]])]
    # sample_data_y = Vector(sample_data[:,"V2"])

    source_data = nothing
    GC.gc()

    mleparm = coef(glm(@formula(V2~V3+V4+V5), sample_data, Binomial()))

    data = [([1;Vector(sample_data[i,["V3", "V4", "V5"]])],sample_data[i,"V2"]) for i in 1:samplesize]


    # gradloglik = (parm,xy) -> xy[1]*(xy[2] - logit(xy[1]'*parm ))
    # gradlogprior = parm -> zeros(parmdim)

    if linkfun == LogitLink()
        invlinkfun = x-> 1/(1+exp(-x))
        dp_dlin = 1 ./(mc_p .* (1 .-mc_p))
        #### I AM HERE
        mc_v = (mc_p .* (1 .-mc_p))
        mc_w = (mc_dp_dlin).^2 ./mc_v
    elseif linkfun == ProbitLink()
        invlinkfun = x-> cdf(Normal(),x)
        mc_p = probit.(mc_data_x*trueparm)
        dp_dlin = exp.(-p.^2/2)/sqrt(2*pi)
        mc_v = (mc_p .* (1 .-mc_p))
        mc_w = (mc_dp_dlin).^2 ./mc_v
    elseif linkfun == CloglogLink()
        invlinkfun = x-> 1-exp(-exp(x))
        mc_p = clogloginv.(mc_data_x*trueparm)
        mc_dp_dlin = -(1 .-mc_p).*log.(1 .-mc_p)
        mc_v = (mc_p .* (1 .-mc_p))
        mc_w = (mc_dp_dlin).^2 ./mc_v
    end

    gradloglik = (parm,xy) ->
    gradlogprior = :flat

    gradloglik_description = "(β,(x,y)) -> x*(y - logit(x'*β ))"
    gradlogprior_description = "β -> zeros(parmdim)"

    ismmap_init = false

    if !initparm
        if mleparmtype == :glm
            initparm = mleparm
        else
            initparm = trueparm
        end
    end

    data_description = "Airline Dataset, Prepocessed by Pollock et al. (2020)"

    @save fname*".jld2" parmdim Jinv V sqrtJinv data trueparm mleparm initparm gradloglik gradlogprior data_description samplesize ismmap_init
end
