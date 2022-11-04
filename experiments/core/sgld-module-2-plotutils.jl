# Computing the IACT
# function autocorrelation(x::AbstractVector, k::Integer, v = var(x))
#     x1 = @view(x[1:(end-k)])
#     x2 = @view(x[(1+k):end])
#     V = sum((x1 .- x2).^2) / length(x1)
#     1 - V / (2*v)
# end
# function iact(x::AbstractVector; v = var(x), maxlag=Inf)
#     N = length(x)
#     τ_inv = 1 + 2 * autocorrelation(x, 1, v)
#     K = 2
#     while K < min(N - 2,maxlag)
#         Δ = autocorrelation(x, K, v) + autocorrelation(x, K + 1, v)
#         if Δ < 0
#             break
#         else
#             τ_inv += 2*Δ
#             K += 2
#         end
#     end
#     τ_inv, K
# end

function autocorrelation(x::AbstractVector)
    T = length(x)
    x_cs = (x .- mean(x)) ./ std(x)
    x_cs2T = [x_cs;zeros(length(x))]
    X_cs2T = rfft(x_cs2T)
    acf = (irfft(abs2.(X_cs2T), 2*T)[1:(T-1)])
    # acf /= acf[1]
    acf = acf ./ ((T-1):-1:1)
    return acf
end
function iact(x::AbstractVector)
    sacf = 0
    T = length(x)
    acf = autocorrelation(x)
    K = 0
    for t in 1:(T-1)
        if acf[t] <0
            break
        end
        sacf += acf[t]
        K += 1
    end
    return real(-1 + 2 * sacf), K
end

# function foldlwhile(op, itr, init, test)
#     output = init
#     i=1
#     while test(itr[i])
#         output = op(output,itr[i])
#         i+=1
#     end
#     output
# end
# function iact(x::AbstractVector; maxlag=Inf)
#     N = length(x)
#     acf = autocor(x,0:min(length(x)-1,maxlag))
#     maxlag = max(length(acf),maxlag)
#     acf = acf[1:maxlag]
#     iact = 2 * foldlwhile(+,acf,0,x-> x>=0) -1
# end


function iact_clean(parm, parmdim, samplesize; extravec1=ones(parmdim)/sqrt(parmdim), extravec2=ones(parmdim)/sqrt(parmdim))
    max(
        round(maximum([iact(map(x->x[i], parm))[1] for i in 1:parmdim])/samplesize,sigdigits=2),
        round(iact(map(x->dot(extravec1,x), parm))[1]/samplesize,sigdigits=2),
        round(iact(map(x->dot(extravec2,x), parm))[1]/samplesize,sigdigits=2)
    )
end
## Custom struct encoding what layers to include for a plot

# the struct
struct Layers
    layerslist::Array{Tuple{Int,Symbol}}
    numlayers::Int
end

# the struct
function Layers()
    (Vector{Tuple{Int,Symbol}}(), 0)
end

# a constructor that lets you list the things you want in a plot by symbol key words
function layerer(components::Vector{Symbol}; drops=[], drop = nothing)
    Layers([(1, c) for c in components], 1)
end

function layerer(component::Symbol; drops=[], drop = nothing)
    Layers([(1, component)], 1)
end

# a method for the constructor that lets add more key words to an existing layer
function layerer(layers::Layers, adds::Vector{Symbol}; drops=[], drop = nothing)
    ll = layers.layerslist
    m = layers.numlayers

    drops = ifelse(drop == nothing, drops, [drops; drop])

    llmp1 = [ll; map(kv -> (m+1,kv[2]), filter(kv -> ((kv[1]==m) & !(kv[2] in drops)), ll)); [(m+1, a) for a in adds]]

    Layers(llmp1,m+1)
end

function layerer(layers::Layers, add::Symbol; drops=[], drop = nothing)
    ll = layers.layerslist
    m = layers.numlayers

    drops = ifelse(drop == nothing, drops, [drops; drop])

    llmp1 = [ll; map(kv -> (m+1,kv[2]), filter(kv -> ((kv[1]==m) & !(kv[2] in drops)), ll)); [(m+1, add)]]

    Layers(llmp1,m+1)
end

# represent the layer as a string of what it contains for filenames
function layer2string(layers::Layers)
    layerstring = ""
    vals = map(kv->kv[2],layers.layerslist)
    for v in vals
        layerstring *= "-"*string(v)
    end
    return layerstring
end

function getsublayers(layers::Layers,k)
    Layers(filter(kv -> kv[1] == k, layers.layerslist),k)
end

## Plotting an SGLD run's density estimator
function sgldplot(fname, samplesize, algos, parms, trueparm, mleparm, Jinv, sqrtJinv, V, w, layers, xlimvals, ylimvals, ismmap_run, numstepss, batchsizes, keepperiods; nameattr= false, levels=10, localscale=true, fontscale=1, legendloc=:topleft, iteravg_numepochs=1)
    parmdim = length(trueparm)
    print("parmdim = "*string(parmdim))
    numlayers = layers.numlayers
    figs = Dict{typeof(numlayers),typeof(plot())}()
    if localscale
        localmap = p -> sqrt(samplesize)*(p .- mleparm)
        invlocalmap = p-> p ./sqrt(samplesize) .+ mleparm
        scalematrix=1
        local_mleparm = zeros(parmdim)
        biv_xlimvals = xlimvals
        biv_ylimvals = ylimvals
    else
        localmap = p -> p
        invlocalmap = p -> p
        biv_xlimvals = xlimvals./sqrt(samplesize) .+ mleparm[1]
        biv_ylimvals = ylimvals./sqrt(samplesize) .+ mleparm[parmdim]
        scalematrix=1/sqrt(samplesize)
        local_mleparm = mleparm
    end

    if !ismmap_run
        local_parms = Dict(algo => localmap.(parm) for (algo,parm) in parms)
    else
        function reader(algo)
            io2 = open(parms[algo][1]*".bin")
            parms_values = Mmap.mmap(io2, parms[algo][2], numstepss[algo]÷keepperiods[algo])
            close(io2)
            # local_parms = localmap.(parms_values[max(1,2*numstepss[algo]÷keepperiods[algo]÷100000):max(1,numstepss[algo]÷keepperiods[algo]÷100000):numstepss[algo]÷keepperiods[algo]])
            localmap.(parms_values)
        end
        local_parms = Dict(algo => reader(algo) for algo in algos)
    end

    local_trueparm = localmap(trueparm)


    # Theoretical densities
    X = range(biv_xlimvals...,length=201)
    Y = range(biv_ylimvals...,length=201)

    # Bivariate
    safe_halfsandwich = (sqrtJinv'*V*sqrtJinv)
    safe_halfsandwich = (safe_halfsandwich +safe_halfsandwich')/2
    safe_halfsandwich = sqrt(safe_halfsandwich*safe_halfsandwich')*scalematrix^2
    f_halfsandwich(x,y) = pdf(MvNormal(local_mleparm[[1,parmdim]], Array(safe_halfsandwich[[1,parmdim],[1,parmdim]])), [x,y])
    Z_halfsandwich = [f_halfsandwich(x,y) for y in Y, x in X ]

    safe_sandwich = Jinv'*V*Jinv
    safe_sandwich = (safe_sandwich+safe_sandwich')/2
    safe_sandwich = sqrt(safe_sandwich*safe_sandwich')*scalematrix^2
    f_sandwich(x,y) = pdf(MvNormal(local_mleparm[[1,parmdim]], Array(safe_sandwich[[1,parmdim],[1,parmdim]])), [x,y])
    Z_sandwich = [f_sandwich(x,y) for y in Y, x in X ]

    safe_Jinv = Jinv
    safe_Jinv = (safe_Jinv+safe_Jinv')/2
    safe_Jinv = sqrt((safe_Jinv)*(safe_Jinv)')*scalematrix^2
    f_Jinv(x,y) = pdf(MvNormal(local_mleparm[[1,parmdim]], Array((safe_Jinv)[[1,parmdim],[1,parmdim]])), [x,y])
    Z_Jinv = [f_Jinv(x,y) for y in Y, x in X ]

    safe_mixture = (w*safe_Jinv + (1-w)*safe_sandwich)
    safe_mixture = (safe_mixture + safe_mixture')/2
    f_mixture(x,y) = pdf(MvNormal(local_mleparm[[1,parmdim]], Array(safe_mixture[[1,parmdim],[1,parmdim]])), [x,y])
    Z_mixture = [f_mixture(x,y) for y in Y, x in X ]

    ### Scaled Bivariate
    safe_halfsandwich_scaled = safe_halfsandwich / iteravg_numepochs
    f_halfsandwich_scaled(x,y) = pdf(MvNormal(local_mleparm[[1,parmdim]], Array(safe_halfsandwich_scaled[[1,parmdim],[1,parmdim]])), [x,y])
    Z_halfsandwich_scaled = [f_halfsandwich_scaled(x,y) for y in Y, x in X ]

    safe_sandwich_scaled = safe_sandwich/ iteravg_numepochs
    f_sandwich_scaled(x,y) = pdf(MvNormal(local_mleparm[[1,parmdim]], Array(safe_sandwich_scaled[[1,parmdim],[1,parmdim]])), [x,y])
    Z_sandwich_scaled = [f_sandwich_scaled(x,y) for y in Y, x in X ]

    safe_Jinv_scaled = safe_Jinv / iteravg_numepochs
    f_Jinv_scaled(x,y) = pdf(MvNormal(local_mleparm[[1,parmdim]], Array((safe_Jinv_scaled)[[1,parmdim],[1,parmdim]])), [x,y])
    Z_Jinv_scaled = [f_Jinv(x,y) for y in Y, x in X ]

    safe_mixture_scaled = safe_mixture / iteravg_numepochs
    f_mixture_scaled(x,y) = pdf(MvNormal(local_mleparm[[1,parmdim]], Array(safe_mixture_scaled[[1,parmdim],[1,parmdim]])), [x,y])
    Z_mixture_scaled = [f_mixture(x,y) for y in Y, x in X ]

    # I wanted to include the corrected (with second order term) for the iteravg covariance,
    # but it requires
    #   1) carry around the precon for each algo (which we dont do now) and
    #   2) knowing Q_infty which we don't always know.

    # sym(A)=0.5*(A+A')
    # prconInv = inv(precon) ### THIS DOESN'T WORK because each algo does not carries arround it's own precon
    # safe_iteravgexact_scaled = safe_sandwich_scaled - 0.5/iteravg_numepochs*sym(...)*scalematrix^2 ### THIS DOESNT WORK in the case where precon=Id because we dont know Q_∞
    # f_iteravgexact_scaled(x,y) = pdf(MvNormal(local_mleparm[[1,parmdim]], Array(safe_iteravgexact_scaled[[1,parmdim],[1,parmdim]])), [x,y])
    # Z_iteravgexact_scaled = [f_iteravgexact_scaled(x,y) for y in Y, x in X ]

    ### KDEs
    kde2d_gen = lp -> InterpKDE(kde(((map(x->x[1], lp), map(x->x[parmdim], lp)))))
    print("Did the runs have any NaNs? If so the KDE estimator will crash\n"*string(Dict(algo => any((x->any(isnan.(x))).(local_parm)) for (algo,local_parm) in local_parms)))
    dens2d = Dict(algo => kde2d_gen(local_parm) for (algo,local_parm) in local_parms)
    Z_kde2d = Dict(algo => [pdf(dens2d[algo],x,y) for y in Y, x in X ] for (algo,local_parm) in local_parms)

    epochlengths = Dict(algo => samplesize ÷ batchsizes[algo] ÷ keepperiods[algo] for algo in algos)
    numepochs = Dict(algo => numstepss[algo]*batchsizes[algo] ÷ samplesize for algo in algos)

    # Univariate
    sublayers1 =  getsublayers(layers,1)
    components1 =  map(kv->kv[2],sublayers1.layerslist)
    isunivariate = (:univariate in components1)
    isallparms = (:allparms in components1)
    ishistogram = (:histogram in components1)
    isiteravg = (:iteravg in components1)
    
    if (isiteravg)
        iteravg_lengths = Dict(algo => epochlengths[algo]*iteravg_numepochs for algo in algos)
        numiteravgs = Dict(algo => numstepss[algo] ÷ iteravg_lengths[algo] for algo in algos)
        # iteravgranges = Dict(algo => [((2*i-1)*(numstepss[algo]÷ keepperiods[algo]÷100)+1):(min((2*i)*(numstepss[algo]÷ keepperiods[algo]÷100), (2*i-1)*(numstepss[algo]÷ keepperiods[algo]÷100)+1*epochlengths[algo])) for i in 1:50] for algo in algos)
        iteravgranges = Dict(algo => [((i-1)*iteravg_lengths[algo]+1):(i*iteravg_lengths[algo]) for i in 1:numiteravgs[algo]] for algo in algos)
        # print(iteravgranges)

        iteravgs = Dict(algo => [sum(local_parm[range])/length(range) for range in iteravgranges[algo]] for (algo,local_parm) in local_parms)
        # print(iteravgs)

        dens2d_avg = Dict(algo => kde2d_gen(iteravg) for (algo,iteravg) in iteravgs)
        Z_kde2d_avg = Dict(algo => [pdf(dens2d_avg[algo],x,y) for y in Y, x in X ] for (algo,iteravg) in iteravgs)
    end
    # iteravgpath = Dict(algo => cumsum(local_parm)./(1:length(local_parm)) for (algo,local_parm) in local_parms)



    if isunivariate
        if !isallparms
            g_halfsandwich_1(x) = pdf(Normal(local_mleparm[1],sqrt(safe_halfsandwich[1,1])), x)
            Z_halfsandwich_1 = [g_halfsandwich_1(x) for x in X ]


            g_sandwich_1(x) = pdf(Normal(local_mleparm[1],sqrt(safe_sandwich[1,1])), x)
            Z_sandwich_1 = [g_sandwich_1(x) for x in X ]

            g_Jinv_1(x) = pdf(Normal(local_mleparm[1],sqrt(safe_Jinv[1,1])), x)
            Z_Jinv_1 = [g_Jinv_1(x) for x in X ]

            g_mixture_1(x) = pdf(Normal(local_mleparm[1],sqrt(safe_mixture[1,1])), x)
            Z_mixture_1 = [g_mixture_1(x) for x in X ]

            g_halfsandwich_scaled_1(x) = pdf(Normal(local_mleparm[1],sqrt(safe_halfsandwich_scaled[1,1])), x)
            Z_halfsandwich_scaled_1 = [g_halfsandwich_scaled_1(x) for x in X ]


            g_sandwich_scaled_1(x) = pdf(Normal(local_mleparm[1],sqrt(safe_sandwich_scaled[1,1])), x)
            Z_sandwich_scaled_1 = [g_sandwich_scaled_1(x) for x in X ]

            g_Jinv_scaled_1(x) = pdf(Normal(local_mleparm[1],sqrt(safe_Jinv_scaled[1,1])), x)
            Z_Jinv_scaled_1 = [g_Jinv_scaled_1(x) for x in X ]

            g_mixture_scaled_1(x) = pdf(Normal(local_mleparm[1],sqrt(safe_mixture_scaled[1,1])), x)
            Z_mixture_scaled_1 = [g_mixture_scaled_1(x) for x in X ]

            kde1d_gen_1 = lp -> kde(map(x->x[1], lp))
            dens1d_1 = Dict(algo => kde1d_gen_1(local_parm) for (algo,local_parm) in local_parms)
        else
            # xlimvals_robust = Dict(p=> (minimum([minimum(map(x->x[p], local_parms[algo])) for algo in algos]) , maximum([maximum(map(x->x[p], local_parms[algo])) for algo in algos])) for p in 1:parmdim)
            if !localscale
                xlimvals_p = Dict(p => xlimvals./sqrt(samplesize) .+ mleparm[p] for p in 1:parmdim)
                Xp =  Dict(p => range(xlimvals_p[p]...,length=201) for p in 1:parmdim)
            else
                xlimvals_p = Dict(p => xlimvals for p in 1:parmdim)
                Xp =  Dict(p => range(xlimvals_p[p]...,length=201) for p in 1:parmdim)
            end

            g_halfsandwich = Dict(p=> x-> pdf(Normal(local_mleparm[p],sqrt(safe_halfsandwich[p,p])), x) for p in 1:parmdim)
            Z_halfsandwich = Dict( p=> [g_halfsandwich[p](x) for x in Xp[p] ] for p in 1:parmdim)

            g_sandwich = Dict(p=> x->pdf(Normal(local_mleparm[p],sqrt(safe_sandwich[p,p])), x) for p in 1:parmdim)
            Z_sandwich = Dict( p=> [g_sandwich[p](x) for x in Xp[p] ] for p in 1:parmdim)

            g_Jinv = Dict(p=> x->pdf(Normal(local_mleparm[p],sqrt(safe_Jinv[p,p])), x) for p in 1:parmdim)
            Z_Jinv = Dict( p=> [g_Jinv[p](x) for x in Xp[p] ] for p in 1:parmdim)

            g_mixture = Dict(p=> x->pdf(Normal(local_mleparm[p],sqrt(safe_mixture[p,p])), x) for p in 1:parmdim)
            Z_mixture = Dict( p=> [g_mixture[p](x) for x in Xp[p] ] for p in 1:parmdim)

            g_halfsandwich_scaled = Dict(p=> x-> pdf(Normal(local_mleparm[p],sqrt(safe_halfsandwich_scaled[p,p])), x) for p in 1:parmdim)
            Z_halfsandwich_scaled = Dict( p=> [g_halfsandwich_scaled[p](x) for x in Xp[p] ] for p in 1:parmdim)

            g_sandwich_scaled = Dict(p=> x->pdf(Normal(local_mleparm[p],sqrt(safe_sandwich_scaled[p,p])), x) for p in 1:parmdim)
            Z_sandwich_scaled = Dict( p=> [g_sandwich_scaled[p](x) for x in Xp[p] ] for p in 1:parmdim)

            g_Jinv_scaled = Dict(p=> x->pdf(Normal(local_mleparm[p],sqrt(safe_Jinv_scaled[p,p])), x) for p in 1:parmdim)
            Z_Jinv_scaled = Dict( p=> [g_Jinv_scaled[p](x) for x in Xp[p] ] for p in 1:parmdim)

            g_mixture_scaled = Dict(p=> x->pdf(Normal(local_mleparm[p],sqrt(safe_mixture_scaled[p,p])), x) for p in 1:parmdim)
            Z_mixture_scaled = Dict( p=> [g_mixture_scaled[p](x) for x in Xp[p] ] for p in 1:parmdim)

            kde1d_gen = Dict(p => (lp -> kde(map(x->x[p], lp))) for p in 1:parmdim)

            dens1d = Dict(p => Dict(algo => kde1d_gen[p](local_parm) for (algo,local_parm) in local_parms) for p in 1:parmdim)
        end
    end







    # local_trueparm = sqrt(samplesize)*(trueparm-mleparm)

    # pval_esgd = round.(ccdf.(Chisq(parmdim), local_trueparm' *((sqrtJinv*V*sqrtJinv)\ local_trueparm)), sigdigits=2)
    # pval_sandwich = round.(ccdf.(Chisq(parmdim), local_trueparm'*( (Jinv*V*Jinv)\ local_trueparm)), sigdigits=2)
    # pval_Jinv = round.(ccdf.(Chisq(parmdim), local_trueparm'* ((Jinv)\ local_trueparm)), sigdigits=2)
    # pval_mix = round.(ccdf.(Chisq(parmdim), local_trueparm'* ((w*Jinv+(1-w)*Jinv*V*Jinv)\ local_trueparm)), sigdigits=2)

    Plots.scalefontsizes(fontscale)
    if !isallparms
        for layernum in 1:numlayers
                sublayers =  getsublayers(layers,layernum)
                components =  map(kv->kv[2],sublayers.layerslist)

                if !(:nolegend in components)
                    figs[layernum] = plot(legend = legendloc)
                else
                    figs[layernum] = plot(legend = false)
                end
                if (:noaxes in components)
                    plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                end
                if (:nogrid in components)
                    plot!(yticks=false,xticks=false)
                end
                # contour plots for theoretical limiting distributions

                if !(:univariate in components)
                    # Bivariate Plots
                        if :halfsandwich in components
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_halfsandwich, seriescolor=cgrad(:blues), linewidth=2 , colorbar = false, linestyle=:dash, levels=levels)
                            plot!(1, color=:blues,linestyle=:dash,label="Basic SGD (expected)")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :sandwich in components
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_sandwich, seriescolor=cgrad(:reds), linewidth=2 , colorbar = false, linestyle=:dash, levels=levels)
                            plot!(1, color=:reds, linestyle=:dash, label="J⁻¹I J⁻¹ Gaussian")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :jinv in components
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_Jinv, seriescolor=cgrad(:PuRd_9, rev=true), linewidth=2 , colorbar = false, linestyle=:dash, levels=levels)
                            plot!(1, color=cgrad(:PuRd_9)[6], linestyle=:dash, label="J⁻¹ Gaussian")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :mixture in components
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_mixture, seriescolor=cgrad(:greens), linewidth=2 , colorbar = false, linestyle=:dash, levels=levels)
                            plot!(1, color=:greens, linestyle=:dash, label="w J⁻¹ + (1-w) J⁻¹I J⁻¹")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :halfsandwichScaled in components
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_halfsandwich_scaled, seriescolor=cgrad(:blues), linewidth=2 , colorbar = false, linestyle=:dash, levels=levels)
                            plot!(1, color=:blues,linestyle=:dash,label="Basic SGD (expected) /m")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :sandwichScaled in components
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_sandwich_scaled, seriescolor=cgrad(:reds), linewidth=2 , colorbar = false, linestyle=:dash, levels=levels)
                            plot!(1, color=:reds, linestyle=:dash, label="J⁻¹I J⁻¹/m Gaussian")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :jinvScaled in components
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_Jinv_scaled, seriescolor=cgrad(:PuRd_9, rev=true), linewidth=2 , colorbar = false, linestyle=:dash, levels=levels)
                            plot!(1, color=cgrad(:PuRd_9)[6], linestyle=:dash, label="J⁻¹/m Gaussian")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :mixtureScaled in components
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_mixture_scaled, seriescolor=cgrad(:greens), linewidth=2 , colorbar = false, linestyle=:dash, levels=levels)
                            plot!(1, color=:greens, linestyle=:dash, label="w J⁻¹/m + (1-w) J⁻¹I J⁻¹/m")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end

                        # true parameter plots with a big star
                        if :trueparm in components
                            # scatter!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, [local_trueparm[1]],[local_trueparm[parmdim]], marker = :star, color=:tan, label = "θ*", markersize = 10)
                            scatter!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, [local_trueparm[1]],[local_trueparm[parmdim]], marker = :star, color=:tan, label = false, markersize = 10)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end

                        # kdes of empirical distributions of sgld runs
                        if !(:iteravg in components)
                        if ((:vanillaSGD  in components))
                            # plot!(dens2d[:vanillaSGD], seriescolor=cgrad(:ice), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d[:vanillaSGD], seriescolor=cgrad(:ice), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:ice)[50], label="Basic SGD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                            # OLD
                            # if (:iteravg in components)
                            #     for i in 1:50
                            #         scatter!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, [iteravgs[:vanillaSGD][i][1]],[iteravgs[:vanillaSGD][i][parmdim]], marker = :diamond, color=:darkblue, label = "Iterate Average", markersize = 5)
                            #         if (:noaxes in components)
                            #             plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            #         end
                            #     end
                            # end
                        end
                        if :preconSGD in components
                            # plot!(dens2d[:preconSGD], seriescolor=cgrad(:amp, rev = true), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d[:preconSGD], seriescolor=cgrad(:amp, rev = true), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:amp)[200], label="J⁻¹-Precon SGD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                            # OLD
                            # if (:iteravg in components)
                            #     for i in 1:50
                            #         scatter!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, [iteravgs[:preconSGD][i][1]],[iteravgs[:preconSGD][i][parmdim]], marker = :diamond, color=:darkblue, label = "Iterate Average", markersize = 5)
                            #         if (:noaxes in components)
                            #             plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            #         end
                            #         if (:nogrid in components)
                            #             plot!(yticks=false,xticks=false)
                            #         end
                            #     end
                            # end
                        end
                        if :preconSGDdiag in components
                            # plot!(dens2d[:preconSGD], seriescolor=cgrad(:amp, rev = true), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d[:preconSGDdiag], seriescolor=cgrad(:amp, rev = true), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:amp)[200], label="diag(J)⁻¹-Precon SGD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                            # OLD
                            # if (:iteravg in components)
                            #     for i in 1:50
                            #         scatter!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, [iteravgs[:preconSGDdiag][i][1]],[iteravgs[:preconSGDdiag][i][parmdim]], marker = :diamond, color=:darkblue, label = "Iterate Average", markersize = 5)
                            #         if (:noaxes in components)
                            #             plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            #         end
                            #         if (:nogrid in components)
                            #             plot!(yticks=false,xticks=false)
                            #         end
                            #     end
                            # end
                        end
                        if :preconVSGD in components
                            # plot!(dens2d[:preconSGD], seriescolor=cgrad(:amp, rev = true), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d[:preconVSGD], seriescolor=cgrad(:amp, rev = true), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:amp)[200], label="I⁻¹-Precon SGD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                            # OLD
                            # if (:iteravg in components)
                            #     for i in 1:50
                            #         scatter!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, [iteravgs[:preconVSGD][i][1]],[iteravgs[:preconVSGD][i][parmdim]], marker = :diamond, color=:darkblue, label = "Iterate Average", markersize = 5)
                            #         if (:noaxes in components)
                            #             plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            #         end
                            #         if (:nogrid in components)
                            #             plot!(yticks=false,xticks=false)
                            #         end
                            #     end
                            # end
                        end
                        if :preconVSGDdiag in components
                            # plot!(dens2d[:preconSGD], seriescolor=cgrad(:amp, rev = true), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d[:preconVSGDdiag], seriescolor=cgrad(:amp, rev = true), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:amp)[200], label="diag(I)⁻¹-Precon SGD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                            # OLD
                            # if (:iteravg in components)
                            #     for i in 1:50
                            #         scatter!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, [iteravgs[:preconVSGDdiag][i][1]],[iteravgs[:preconVSGDdiag][i][parmdim]], marker = :diamond, color=:darkblue, label = "Iterate Average", markersize = 5)
                            #         if (:noaxes in components)
                            #             plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            #         end
                            #         if (:nogrid in components)
                            #             plot!(yticks=false,xticks=false)
                            #         end
                            #     end
                            # end
                        end
                        if :vanillaSGLD in components
                            # plot!(dens2d[:vanillaSGLD], seriescolor=cgrad(:acton), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d[:vanillaSGLD], seriescolor=cgrad(:acton), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:acton)[50], label="Basic SGLD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                            # OLD
                            # if (:iteravg in components)
                            #     for i in 1:50
                            #         scatter!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, [iteravgs[:vanillaSGLD][i][1]],[iteravgs[:vanillaSGLD][i][parmdim]], marker = :diamond, color=:darkblue, label = "Iterate Average", markersize = 5)
                            #         if (:noaxes in components)
                            #             plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            #         end
                            #         if (:nogrid in components)
                            #             plot!(yticks=false,xticks=false)
                            #         end
                            #     end
                            # end
                        end
                        if :preconSGLD in components
                            # plot!(dens2d[:preconSGLD], seriescolor=cgrad(:algae, rev=true), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d[:preconSGLD], seriescolor=cgrad(:algae, rev=true), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:algae)[200], label="J⁻¹-Precon SGLD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                            # OLD
                            # if (:iteravg in components)
                            #     for i in 1:50
                            #         scatter!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, [iteravgs[:preconSGLD][i][1]],[iteravgs[:preconSGLD][i][parmdim]], marker = :diamond, color=:darkblue, label = "Iterate Average", markersize = 5)
                            #         if (:noaxes in components)
                            #             plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            #         end
                            #         if (:nogrid in components)
                            #             plot!(yticks=false,xticks=false)
                            #         end
                            #     end
                            # end
                        end
                        if :preconSGLDdiag in components
                            # plot!(dens2d[:preconSGLD], seriescolor=cgrad(:algae, rev=true), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d[:preconSGLDdiag], seriescolor=cgrad(:algae, rev=true), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:algae)[200], label="diag(J)⁻¹-Precon SGLD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                            # OLD
                            # if (:iteravg in components)
                            #     for i in 1:50
                            #         scatter!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, [iteravgs[:preconSGLDdiag][i][1]],[iteravgs[:preconSGLDdiag][i][parmdim]], marker = :diamond, color=:darkblue, label = "Iterate Average", markersize = 5)
                            #         if (:noaxes in components)
                            #             plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            #         end
                            #         if (:nogrid in components)
                            #             plot!(yticks=false,xticks=false)
                            #         end
                            #     end
                            # end
                        end
                        if :largeBatchVanillaSGLD in components
                            # plot!(dens2d[:largeBatchVanillaSGLD], seriescolor=cgrad(:BuPu_9, rev=true), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d[:largeBatchVanillaSGLD], seriescolor=cgrad(:BuPu_9, rev=true), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:BuPu_9, rev=true)[6], label="Large Batch SGLD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :largeBatchPreconSGLD in components
                            # plot!(dens2d[:largeBatchPreconSGLD], seriescolor=cgrad(:BuPu_9, rev=true), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d[:largeBatchPreconSGLD], seriescolor=cgrad(:BuPu_9, rev=true), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:BuPu_9)[6], label="Large Batch Preconditioned SGLD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :vanillaSGLDdecreasing in components
                            # plot!(dens2d[:vanillaSGLDdecreasing], seriescolor=cgrad(:BuPu_9, rev=true), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d[:vanillaSGLDdecreasing], seriescolor=cgrad(:BuPu_9, rev=true), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:BuPu_9)[6], label="SGLD with decreasing step sizes")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end

                        ### ITERAVG FOR POWERS
                        if :vanillaSGDPow  in components
                            if (:iteravg in components)
                                for i in 1:50
                                    scatter!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, [iteravgs[:vanillaSGDPow][i][1]],[iteravgs[:vanillaSGDPow][i][parmdim]], marker = :diamond, color=:darkblue, label = false, markersize = 5)
                                    if (:noaxes in components)
                                        plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                                    end
                                    if (:nogrid in components)
                                        plot!(yticks=false,xticks=false)
                                    end
                                end
                            end

                        end

                        if :vanillaSGLDPow in components
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                            if (:iteravg in components)
                                for i in 1:50
                                    scatter!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, [iteravgs[:vanillaSGLDPow][i][1]],[iteravgs[:vanillaSGLDPow][i][parmdim]], marker = :diamond, color=:darkred, label = false, markersize = 5)
                                    if (:noaxes in components)
                                        plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                                    end
                                    if (:nogrid in components)
                                        plot!(yticks=false,xticks=false)
                                    end
                                end
                            end
                        end
                        ####### New ITER AVG
                        if ((:vanillaSGDPowHalf in components))
                            # plot!(dens2d[:vanillaSGD], seriescolor=cgrad(:ice), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d[:vanillaSGDPowHalf], seriescolor=cgrad(:ice), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:ice)[50], label="Basic SGD, power=1/2")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end

                        if ((:vanillaSGDPowThird in components))
                            # plot!(dens2d[:vanillaSGD], seriescolor=cgrad(:ice), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d[:vanillaSGDPowThird], seriescolor=cgrad(:algae, rev=true), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:ice)[50], label="Basic SGD, power=1/3")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if ((:vanillaSGDPowTwoThird in components))
                            # plot!(dens2d[:vanillaSGD], seriescolor=cgrad(:ice), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d[:vanillaSGDPowTwoThird], seriescolor=cgrad(:algae, rev=true), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:ice)[50], label="Basic SGD, power=2/3")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if ((:vanillaSGDPowHalfSumOne in components))
                            # plot!(dens2d[:vanillaSGD], seriescolor=cgrad(:ice), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d[:vanillaSGDPowHalfSumOne], seriescolor=cgrad(:ice), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:ice)[50], label="Basic SGD, power=1/2")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end

                        if ((:vanillaSGDPowThirdSumOne in components))
                            # plot!(dens2d[:vanillaSGD], seriescolor=cgrad(:ice), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d[:vanillaSGDPowThirdSumOne], seriescolor=cgrad(:algae, rev=true), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:ice)[50], label="Basic SGD, power=1/3")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if ((:vanillaSGDPowTwoThirdSumOne in components))
                            # plot!(dens2d[:vanillaSGD], seriescolor=cgrad(:ice), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d[:vanillaSGDPowTwoThirdSumOne], seriescolor=cgrad(:algae, rev=true), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:ice)[50], label="Basic SGD, power=2/3")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if ((:vanillaSGDPowOneSumOne in components))
                            # plot!(dens2d[:vanillaSGD], seriescolor=cgrad(:ice), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d[:vanillaSGDPowOneSumOne], seriescolor=cgrad(:algae, rev=true), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:ice)[50], label="Basic SGD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        plot!(xlims=biv_xlimvals, ylims=biv_ylimvals)
                    end
                    if (:iteravg in components)
                        if ((:vanillaSGDPowHalf in components) & (:iteravg in components))
                            # plot!(dens2d[:vanillaSGD], seriescolor=cgrad(:ice), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d_avg[:vanillaSGDPowHalf], seriescolor=cgrad(:ice), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:ice)[50], label="Basic SGD Iter. Avg., power=1/2")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if ((:vanillaSGDPowThird in components) & (:iteravg in components))
                            # plot!(dens2d[:vanillaSGD], seriescolor=cgrad(:ice), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d_avg[:vanillaSGDPowThird], seriescolor=cgrad(:algae, rev=true), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:ice)[50], label="Basic SGD Iter. Avg., power=1/3")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if ((:vanillaSGDPowTwoThird in components) & (:iteravg in components))
                            # plot!(dens2d[:vanillaSGD], seriescolor=cgrad(:ice), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d_avg[:vanillaSGDPowTwoThird], seriescolor=cgrad(:algae, rev=true), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:ice)[50], label="Basic SGD Iter. Avg., power=2/3")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if ((:vanillaSGDPowOne in components))
                            # plot!(dens2d[:vanillaSGD], seriescolor=cgrad(:ice), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d_avg[:vanillaSGDPowOne], seriescolor=cgrad(:ice), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:ice)[50], label="Basic SGD Iter. Avg.")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if ((:vanillaSGDPowHalfSumOne in components) & (:iteravg in components))
                            # plot!(dens2d[:vanillaSGD], seriescolor=cgrad(:ice), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d_avg[:vanillaSGDPowHalfSumOne], seriescolor=cgrad(:ice), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:ice)[50], label="Basic SGD Iter. Avg., power=1/2")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if ((:vanillaSGDPowThirdSumOne in components) & (:iteravg in components))
                            # plot!(dens2d[:vanillaSGD], seriescolor=cgrad(:ice), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d_avg[:vanillaSGDPowThirdSumOne], seriescolor=cgrad(:algae, rev=true), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:ice)[50], label="Basic SGD Iter. Avg., power=1/3")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if ((:vanillaSGDPowTwoThirdSumOne in components) & (:iteravg in components))
                            # plot!(dens2d[:vanillaSGD], seriescolor=cgrad(:ice), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d_avg[:vanillaSGDPowTwoThirdSumOne], seriescolor=cgrad(:algae, rev=true), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:ice)[50], label="Basic SGD Iter. Avg., power=2/3")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if ((:vanillaSGDPowOneSumOne in components))
                            # plot!(dens2d[:vanillaSGD], seriescolor=cgrad(:ice), linewidth=0.9, levels=levels)
                            contour!(twinx(), yaxis=false, yticks=false, zaxis=false,zticks=false, X,Y,Z_kde2d_avg[:vanillaSGDPowOneSumOne], seriescolor=cgrad(:ice), linewidth=0.9 , colorbar = false, levels=levels)
                            plot!(1, color=cgrad(:ice)[50], label="Basic SGD Iter. Avg.")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        plot!(xlims=biv_xlimvals./sqrt(iteravg_numepochs), ylims=biv_ylimvals./sqrt(iteravg_numepochs))
                    end


                else
                    # Univariate Plots
                    if !ishistogram
                        if :halfsandwich in components
                            plot!(X,Z_halfsandwich_1, seriescolor=:blue, linewidth=2 , linestyle=:dash,label="Basic SGD (expected)")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :sandwich in components
                            plot!(X,Z_sandwich_1, seriescolor=:red, linewidth=2 , linestyle=:dash, label="J⁻¹I J⁻¹ Gaussian")
                        end
                        if :jinv in components
                            plot!(X,Z_Jinv_1, seriescolor=:purple, linewidth=2 ,  linestyle=:dash, label="J⁻¹ Gaussian")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :mixture in components
                            plot!(X,Z_mixture_1, seriescolor=:green, linewidth=2 , linestyle=:dash, label="w J⁻¹ + (1-w) J⁻¹I J⁻¹")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :halfsandwichScaled in components
                            plot!(X,Z_halfsandwich_scaled_1, seriescolor=:blue, linewidth=2 , linestyle=:dash,label="Basic SGD (expected)/m")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :sandwichScaled in components
                            plot!(X,Z_sandwich_scaled_1, seriescolor=:red, linewidth=2 , linestyle=:dash, label="J⁻¹I J⁻¹/m Gaussian")
                        end
                        if :jinvScaled in components
                            plot!(X,Z_Jinv_scaled_1, seriescolor=:purple, linewidth=2 ,  linestyle=:dash, label="J⁻/m¹ Gaussian")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :mixtureScaled in components
                            plot!(X,Z_mixture_scaled_1, seriescolor=:green, linewidth=2 , linestyle=:dash, label="w J⁻¹/m + (1-w) J⁻¹I J⁻¹/m")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end

                        # true parameter plots with a big star
                        if :trueparm in components
                            # vline!([local_trueparm[1]], color=:tan, label = "θ*")
                            vline!([local_trueparm[1]], color=:tan, label = "True Parameter", linewidth=4)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end

                        # kdes of empirical distributions of sgld runs
                        if :vanillaSGD  in components
                            plot!(dens1d_1[:vanillaSGD], seriescolor=:turquoise3, linewidth=0.9, label="Basic SGD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconSGD in components
                            plot!(dens1d_1[:preconSGD], seriescolor=:orange, linewidth=0.9, label="J⁻¹-Precon SGD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconSGDdiag in components
                            plot!(dens1d_1[:preconSGDdiag], seriescolor=:orange, linewidth=0.9, label="diag(J)⁻¹-Precon SGD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconVSGD in components
                            plot!(dens1d_1[:preconVSGD], seriescolor=:orange, linewidth=0.9, label="I⁻¹-Precon SGD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconVSGDdiag in components
                            plot!(dens1d_1[:preconVSGDdiag], seriescolor=:orange, linewidth=0.9, label="diag(I)⁻¹-Precon SGD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :vanillaSGLD in components
                            plot!(dens1d_1[:vanillaSGLD], seriescolor=:orange, linewidth=0.9,  label="Basic SGLD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconSGLD in components
                            plot!(dens1d_1[:preconSGLD], seriescolor=:blue, linewidth=0.9,  label="J⁻¹-Precon SGLD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconSGLDdiag in components
                            plot!(dens1d_1[:preconSGLDdiag], seriescolor=:blue, linewidth=0.9,  label="diag(J)⁻¹-Precon SGLD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :largeBatchVanillaSGLD in components
                            plot!(dens1d_1[:largeBatchVanillaSGLD], seriescolor=:magenta, linewidth=0.9, label="Large Sample SGLD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :largeBatchPreconSGLD in components
                            plot!(dens1d_1[:largeBatchPreconSGLD], seriescolor=:magenta, linewidth=0.9, label="Large Sample Preconditioned SGLD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :vanillaSGLDdecreasing in components
                            plot!(dens1d_1[:vanillaSGLDdecreasing], seriescolor=:magenta, linewidth=0.9, label="SGLD with decreasing step sizes")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end

                        ### ITERAVG FOR POWERS
                        if :vanillaSGDPow  in components
                            if (:iteravg in components)
                                for i in 1:50
                                    vline!([iteravgs[:vanillaSGDPow][i][1]], color=:darkblue, label = false, linewidth=1)
                                end
                            end
                        end

                        if :vanillaSGLDPow in components
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                            if (:iteravg in components)
                                for i in 1:50
                                    vline!([iteravgs[:vanillaSGLDPow][i][1]], color=:darkred, label = false, linewidth=1)
                                end
                            end
                        end


                        plot!(xlims=biv_xlimvals,yticks=false)
                        if (:noaxes in components)
                            plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                        end
                        if (:nogrid in components)
                            plot!(yticks=false,xticks=false)
                        end
                    else
                        if :halfsandwich in components
                            plot!(X,Z_halfsandwich_1, seriescolor=:blue, linewidth=4 , label="Basic SGD (expected)")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :sandwich in components
                            plot!(X,Z_sandwich_1, seriescolor=:red, linewidth=4 , label="J⁻¹I J⁻¹ Gaussian")
                        end
                        if :jinv in components
                            plot!(X,Z_Jinv_1, seriescolor=:purple, linewidth=4 ,  label="J⁻¹ Gaussian")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :mixture in components
                            plot!(X,Z_mixture_1, seriescolor=:green, linewidth=4 , label="w J⁻¹ + (1-w) J⁻¹I J⁻¹")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :halfsandwichScaled in components
                            plot!(X,Z_halfsandwich_scaled_1, seriescolor=:blue, linewidth=4 , label="Basic SGD (expected)")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :sandwichScaled in components
                            plot!(X,Z_sandwich_scaled_1, seriescolor=:red, linewidth=4 , label="J⁻¹I J⁻¹/m Gaussian")
                        end
                        if :jinvScaled in components
                            plot!(X,Z_Jinv_scaled_1, seriescolor=:purple, linewidth=4 ,  label="J⁻¹/m Gaussian")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :mixtureScaled in components
                            plot!(X,Z_mixture_scaled_1, seriescolor=:green, linewidth=4 , label="w J⁻¹/m + (1-w) J⁻¹I J⁻¹/m")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end

                        # true parameter plots with a big star
                        if :trueparm in components
                            # vline!([local_trueparm[1]], color=:tan, label = "θ*")
                            vline!([local_trueparm[1]], color=:tan, label = "True Parameter", linewidth=4)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end

                        # kdes of empirical distributions of sgld runs
                        if :vanillaSGD  in components
                            histogram!(map(x->x[1], local_parms[:vanillaSGD]), alpha=0.25, color = :turquoise3, linecolor = :match,label="Basic SGD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconSGD in components
                            histogram!(map(x->x[1], local_parms[:preconSGD]), alpha=0.25, color = :orange,linecolor = :match,label="J⁻¹-Precon SGD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconSGDdiag in components
                            histogram!(map(x->x[1], local_parms[:preconSGDdiag]), alpha=0.25, color = :orange,linecolor = :match,label="diag(J)⁻¹-Precon SGD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconVSGD in components
                            histogram!(map(x->x[1], local_parms[:preconVSGD]), alpha=0.25, color = :orange,linecolor = :match,label="I⁻¹-Precon SGD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconVSGDdiag in components
                            histogram!(map(x->x[1], local_parms[:preconVSGDdiag]), alpha=0.25, color = :orange,linecolor = :match,label="diag(I)⁻¹-Precon SGD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :vanillaSGLD in components
                            histogram!(map(x->x[1], local_parms[:vanillaSGLD]), alpha=0.25, color = :orange,linecolor = :match,label="Basic SGLD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconSGLD in components
                            histogram!(map(x->x[1], local_parms[:preconSGLD]), alpha=0.25, color = :blue,linecolor = :match,label="J⁻¹-Precon SGLD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                            if (:iteravg in components)
                                vline!([iteravgs[:preconSGLD][1]], color=:green2, label = false)
                            end
                        end
                        if :preconSGLDdiag in components
                            histogram!(map(x->x[1], local_parms[:preconSGLDdiag]), alpha=0.25, color = :blue,linecolor = :match,label="diag(J)⁻¹-Precon SGLD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                            if (:iteravg in components)
                                vline!([iteravgs[:preconSGLDdiag][1]], color=:green2, label = false)
                            end
                        end
                        if :largeBatchVanillaSGLD in components
                            histogram!(map(x->x[1], local_parms[:largeBatchVanillaSGLD]), alpha=0.25, color = :magenta,linecolor = :match,label="Large Sample SGLD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :largeBatchPreconSGLD in components
                            histogram!(map(x->x[1], local_parms[:largeBatchPreconSGLD]), alpha=0.25, color = :magenta,linecolor = :match,label="Large Sample Preconditioned SGLD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :vanillaSGLDdecreasing in components
                            histogram!(map(x->x[1], local_parms[:vanillaSGLDdecreasing]), alpha=0.25, color = :magenta,linecolor = :match,label="SGLD with decreasing step sizes",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end

                        ### ITERAVG FOR POWERS
                        if :vanillaSGDPow  in components
                            if (:iteravg in components)
                                for i in 1:50
                                    vline!([iteravgs[:vanillaSGDPow][i][1]], color=:darkblue, label = false, linewidth=1)
                                end
                            end
                        end

                        if :vanillaSGLDPow in components
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                            if (:iteravg in components)
                                for i in 1:50
                                    vline!([iteravgs[:vanillaSGLDPow][i][1]], color=:darkred, label = false, linewidth=1)
                                end
                            end
                        end

                        plot!(xlims=biv_xlimvals,yticks=false)
                        if (:noaxes in components)
                            plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                        end
                        if (:nogrid in components)
                            plot!(yticks=false,xticks=false)
                        end
                    end
                end
                # Plots.scalefontsizes(1/1.5)
                plot!(foreground_color_legend = nothing, background_color_legend = nothing)
                if !(:nolegend in components)
                    figs[layernum] = plot!(legend = legendloc)
                else
                    figs[layernum] = plot!(legend = false)
                end

                if nameattr
                    savefig(figs[layernum], fname*layer2string(sublayers)*".png")
                    savefig(figs[layernum], fname*layer2string(sublayers)*".pdf")
                else
                    savefig(figs[layernum], fname*"-layer-"*string(layernum)*".png")
                    savefig(figs[layernum], fname*"-layer-"*string(layernum)*".pdf")
                end
        end
    else
        for p in 1:parmdim
            for layernum in 1:numlayers
                    sublayers =  getsublayers(layers,layernum)
                    components =  map(kv->kv[2],sublayers.layerslist)

                    if !(:nolegend in components)
                        figs[layernum] = plot(legend = legendloc)
                    else
                        figs[layernum] = plot(legend = false)
                    end
                    if (:noaxes in components)
                        plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                    end
                    if (:nogrid in components)
                        plot!(yticks=false,xticks=false)
                    end

                    # Univariate Plots
                    if !ishistogram
                        if :halfsandwich in components
                            plot!(Xp[p],Z_halfsandwich[p], seriescolor=:blue, linewidth=2 , linestyle=:dash,label="Basic SGD (expected)")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :sandwich in components
                            plot!(Xp[p],Z_sandwich[p], seriescolor=:red, linewidth=2 , linestyle=:dash, label="J⁻¹I J⁻¹ Gaussian")
                        end
                        if :jinv in components
                            plot!(Xp[p],Z_Jinv[p], seriescolor=:purple, linewidth=2 ,  linestyle=:dash, label="J⁻¹ Gaussian")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :mixture in components
                            plot!(Xp[p],Z_mixture[p], seriescolor=:green, linewidth=2 , linestyle=:dash, label="w J⁻¹ + (1-w) J⁻¹I J⁻¹")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :halfsandwichScaled in components
                            plot!(Xp[p],Z_halfsandwich_scaled[p], seriescolor=:blue, linewidth=2 , linestyle=:dash,label="Basic SGD (expected/m)")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :sandwichScaled in components
                            plot!(Xp[p],Z_sandwich_scaled[p], seriescolor=:red, linewidth=2 , linestyle=:dash, label="J⁻¹I J⁻¹/m Gaussian")
                        end
                        if :jinvScaled in components
                            plot!(Xp[p],Z_Jinv_scaled[p], seriescolor=:purple, linewidth=2 ,  linestyle=:dash, label="J⁻¹/m Gaussian")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :mixtureScaled in components
                            plot!(Xp[p],Z_mixture_scaled[p], seriescolor=:green, linewidth=2 , linestyle=:dash, label="w J⁻¹/m + (1-w) J⁻¹I J⁻¹/m")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end

                        # true parameter plots with a big star
                        if :trueparm in components
                            # vline!([local_trueparm[p]], color=:tan, label = "θ*")
                            vline!([local_trueparm[p]], color=:tan, label = false, linewidth=4)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end

                        # kdes of empirical distributions of sgld runs
                        if :vanillaSGD  in components
                            plot!(dens1d[p][:vanillaSGD], seriescolor=:turquoise3, linewidth=0.9, label="Basic SGD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconSGD in components
                            plot!(dens1d[p][:preconSGD], seriescolor=:orange, linewidth=0.9, label="J⁻¹-Precon SGD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconSGDdiag in components
                            plot!(dens1d[p][:preconSGDdiag], seriescolor=:orange, linewidth=0.9, label="diag(J)⁻¹-Precon SGD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconVSGD in components
                            plot!(dens1d[p][:preconVSGD], seriescolor=:orange, linewidth=0.9, label="I⁻¹-Precon SGD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconVSGDdiag in components
                            plot!(dens1d[p][:preconVSGDdiag], seriescolor=:orange, linewidth=0.9, label="diag(I)⁻¹-Precon SGD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :vanillaSGLD in components
                            plot!(dens1d[p][:vanillaSGLD], seriescolor=:orange, linewidth=0.9,  label="Basic SGLD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconSGLD in components
                            plot!(dens1d[p][:preconSGLD], seriescolor=:blue, linewidth=0.9,  label="J⁻¹-Precon SGLD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconSGLDdiag in components
                            plot!(dens1d[p][:preconSGLDdiag], seriescolor=:blue, linewidth=0.9,  label="diag(J)⁻¹-Precon SGLD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :largeBatchVanillaSGLD in components
                            plot!(dens1d[p][:largeBatchVanillaSGLD], seriescolor=:magenta, linewidth=0.9, label="Large Sample SGLD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :largeBatchPreconSGLD in components
                            plot!(dens1d[p][:largeBatchPreconSGLD], seriescolor=:magenta, linewidth=0.9, label="Large Sample Preconditioned SGLD")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :vanillaSGLDdecreasing in components
                            plot!(dens1d[p][:vanillaSGLDdecreasing], seriescolor=:magenta, linewidth=0.9, label="SGLD with decreasing step sizes")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        plot!(xlims=xlimvals_p[p],yticks=false)
                        if (:noaxes in components)
                            plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                        end
                        if (:nogrid in components)
                            plot!(yticks=false,xticks=false)
                        end
                    else
                        if :halfsandwich in components
                            plot!(Xp[p],Z_halfsandwich[p], seriescolor=:blue, linewidth=4 , label="Basic SGD (expected)")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :sandwich in components
                            plot!(Xp[p],Z_sandwich[p], seriescolor=:red, linewidth=4 , label="J⁻¹I J⁻¹ Gaussian")
                        end
                        if :jinv in components
                            plot!(Xp[p],Z_Jinv[p], seriescolor=:purple, linewidth=4 ,  label="J⁻¹ Gaussian")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :mixture in components
                            plot!(Xp[p],Z_mixture[p], seriescolor=:green, linewidth=4 , label="w J⁻¹ + (1-w) J⁻¹I J⁻¹")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :halfsandwichScaled in components
                            plot!(Xp[p],Z_halfsandwich_scaled[p], seriescolor=:blue, linewidth=4 , label="Basic SGD (expected)/m")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :sandwichScaled in components
                            plot!(Xp[p],Z_sandwich_scaled[p], seriescolor=:red, linewidth=4 , label="J⁻¹I J⁻¹ Gaussian/m")
                        end
                        if :jinvScaled in components
                            plot!(Xp[p],Z_Jinv_scaled[p], seriescolor=:purple, linewidth=4 ,  label="J⁻¹/m Gaussian")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :mixtureScaled in components
                            plot!(Xp[p],Z_mixture_scaled[p], seriescolor=:green, linewidth=4 , label="w J⁻¹/m + (1-w) J⁻¹I J⁻¹/m")
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end

                        # true parameter plots with a big star
                        if :trueparm in components
                            # vline!([local_trueparm[p]], color=:tan, label = "θ*")
                            vline!([local_trueparm[p]], color=:black, label = "θ*", linewidth=4)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end

                        # kdes of empirical distributions of sgld runs
                        if :vanillaSGD  in components
                            histogram!(map(x->x[p], local_parms[:vanillaSGD]), alpha=0.25, color = :turquoise3,linecolor = :match,label="Basic SGD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconSGD in components
                            histogram!(map(x->x[p], local_parms[:preconSGD]), alpha=0.25, color = :orange,linecolor = :match,label="J⁻¹-Precon SGD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconSGDdiag in components
                            histogram!(map(x->x[p], local_parms[:preconSGDdiag]), alpha=0.25, color = :orange,linecolor = :match,label="diag(J)⁻¹-Precon SGD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconVSGD in components
                            histogram!(map(x->x[p], local_parms[:preconVSGD]), alpha=0.25, color = :orange,linecolor = :match,label="I⁻¹-Precon SGD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconVSGDdiag in components
                            histogram!(map(x->x[p], local_parms[:preconVSGDdiag]), alpha=0.25, color = :orange,linecolor = :match,label="diag(I)⁻¹-Precon SGD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :vanillaSGLD in components
                            histogram!(map(x->x[p], local_parms[:vanillaSGLD]), alpha=0.25, color = :orange,linecolor = :match,label="Basic SGLD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconSGLD in components
                            histogram!(map(x->x[p], local_parms[:preconSGLD]), alpha=0.25, color = :blue,linecolor = :match,label="J⁻¹-Precon SGLD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :preconSGLDdiag in components
                            histogram!(map(x->x[p], local_parms[:preconSGLDdiag]), alpha=0.25, color = :blue,linecolor = :match,label="diag(J)⁻¹-Precon SGLD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end

                        if :largeBatchVanillaSGLD in components
                            histogram!(map(x->x[p], local_parms[:largeBatchVanillaSGLD]), alpha=0.25, color = :magenta, linecolor = :match,label="Large Sample SGLD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :largeBatchPreconSGLD in components
                            histogram!(map(x->x[p], local_parms[:largeBatchPreconSGLD]), alpha=0.25, color = :magenta, linecolor = :match,label="Large Sample Preconditioned SGLD",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        if :vanillaSGLDdecreasing in components
                            histogram!(map(x->x[p], local_parms[:vanillaSGLDdecreasing]), alpha=0.25, color = :magenta, linecolor = :match,label="SGLD with decreasing step sizes",normalize=true)
                            if (:noaxes in components)
                                plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                            end
                            if (:nogrid in components)
                                plot!(yticks=false,xticks=false)
                            end
                        end
                        plot!(xlims=xlimvals_p[p],yticks=false,ylims=1.25.*ylims())
                        if (:noaxes in components)
                            plot!(yaxis=false,xaxis=false,yticks=false,xticks=false)
                        end
                        if (:nogrid in components)
                            plot!(yticks=false,xticks=false)
                        end
                    end
                    plot!(foreground_color_legend = nothing, background_color_legend = nothing)
                    if nameattr
                        savefig(figs[layernum], fname*"-parameter-"*string(p)*"-"*layer2string(sublayers)*".png")
                        savefig(figs[layernum], fname*"parameter-"*string(p)*"-"*layer2string(sublayers)*".pdf")
                    else
                        savefig(figs[layernum], fname*"-parameter-"*string(p)*"-layer-"*string(layernum)*".png")
                        savefig(figs[layernum], fname*"-parameter-"*string(p)*"-layer-"*string(layernum)*".pdf")
                    end
            end
        end
    end
    Plots.scalefontsizes(1/fontscale)
end
