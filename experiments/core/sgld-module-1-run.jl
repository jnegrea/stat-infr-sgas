## One-step update funciton
function sgld_update_1step(parm, batchdata, gradloglik, gradlogprior,
    samplesize, batchsize, parmdim;
    stepsize=1, invtemp=false, precon=false, sqrtaniso=false)

    precon == false && (precon=1)
    sqrtaniso == false && (sqrtaniso=1)

    if invtemp==false
        incrGaussian = 0
    else
        if typeof(parm) <: Union{Array, SArray}
            incrGaussian = sqrt(stepsize/invtemp) * sqrtaniso * randn(parmdim)
        else
            incrGaussian = sqrt(stepsize/invtemp) * sqrtaniso * randn()
        end
    end

    incrPrior = stepsize/2 * precon * gradlogprior(parm)

    currygradloglik(x) = precon * gradloglik(parm, x)

    incrLoglik = stepsize/2 * samplesize/batchsize * sum(currygradloglik.(batchdata))

    return parm .+ incrGaussian .+ incrPrior .+ incrLoglik
end

## Main function for running SGLD
function sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
    parmdim, numsteps, ismmap_init, ismmap_run; stepsize=1, invtemp=false, precon=false, sqrtaniso=false, replacement= true, fname="sgldrun", writeperiod=10000, power=0, keepperiod=1)

    if keepperiod ==1
        if (!ismmap_init)&(!ismmap_run)
            dataset = data
            parms = Array{typeof(initparm),1}(undef, numsteps)
            parms[1]= initparm

            @showprogress for iternum in 2:numsteps
                batchdata = dataset[sample(1:samplesize, batchsize; replace = replacement),:]
                parms[iternum] = sgld_update_1step(parms[iternum-1], batchdata, gradloglik, gradlogprior,
                    samplesize, batchsize, parmdim;
                    stepsize=stepsize, invtemp=invtemp, precon=precon, sqrtaniso=sqrtaniso)
            end
            return parms
        elseif ismmap_init&ismmap_run
            io = open(data[1]*".bin")
            dataset = Mmap.mmap(io,data[2],samplesize)
            # close(io)

            io2 = open(fname*"-run.bin", "w+")
            parmstype = Vector{SVector{parmdim,Float64}}
            parms = Mmap.mmap(io2, parmstype, numsteps)
            parms[1]= initparm
            Mmap.sync!(parms)

            @showprogress for iternum in 2:numsteps
                batchdata = dataset[sample(1:samplesize, batchsize; replace = replacement),:]
                parms[iternum] = sgld_update_1step(parms[iternum-1], batchdata, gradloglik, gradlogprior,
                    samplesize, batchsize, parmdim;
                    stepsize=stepsize/(1+iternum)^power, invtemp=invtemp, precon=precon, sqrtaniso=sqrtaniso)
                if (iternum % writeperiod ==0)
                    Mmap.sync!(parms)
                end
            end
            Mmap.sync!(parms)
            close(io)
            close(io2)
            dataset = nothing
            parms = nothing
            return (data[1]*"-"*fname, parmstype)
        elseif (!ismmap_init)&ismmap_run
            dataset = data
            io2 = open(fname*"-run.bin", "w+")
            parmstype = Vector{SVector{parmdim,Float64}}
            parms = Mmap.mmap(io2, parmstype, numsteps)
            parms[1]= initparm
            Mmap.sync!(parms)

            @showprogress for iternum in 2:numsteps
                batchdata = dataset[sample(1:samplesize, batchsize; replace = replacement),:]
                parms[iternum] = sgld_update_1step(parms[iternum-1], batchdata, gradloglik, gradlogprior,
                    samplesize, batchsize, parmdim;
                    stepsize=stepsize/(1+iternum)^power, invtemp=invtemp, precon=precon, sqrtaniso=sqrtaniso)
                if (iternum % writeperiod ==0)
                    Mmap.sync!(parms)
                end
            end
            Mmap.sync!(parms)
            close(io2)
            dataset = nothing
            parms = nothing
            return (fname*"-run", parmstype)
        elseif (ismmap_init)&(!ismmap_run)
            io = open(data[1]*".bin")
            dataset = Mmap.mmap(io,data[2],samplesize)

            parms = Array{typeof(initparm),1}(undef, numsteps)
            parms[1]= initparm

            @showprogress for iternum in 2:numsteps
                batchdata = dataset[sample(1:samplesize, batchsize; replace = replacement),:]
                parms[iternum] = sgld_update_1step(parms[iternum-1], batchdata, gradloglik, gradlogprior,
                    samplesize, batchsize, parmdim;
                    stepsize=stepsize, invtemp=invtemp, precon=precon, sqrtaniso=sqrtaniso)
            end
            return parms
        else
            error("invalid ismmap parameters")
        end
    else
        if (!ismmap_init)&(!ismmap_run)
            dataset = data
            parms = Array{typeof(initparm),1}(undef, numsteps÷keepperiod)
            parms[1] = initparm
            i = 1
            tempparm = initparm

            @showprogress for iternum in 2:numsteps
                batchdata = dataset[sample(1:samplesize, batchsize; replace = replacement),:]
                tempparm = sgld_update_1step(tempparm, batchdata, gradloglik, gradlogprior,
                    samplesize, batchsize, parmdim;
                    stepsize=stepsize, invtemp=invtemp, precon=precon, sqrtaniso=sqrtaniso)
                if (iternum % keepperiod == 1)
                    i += 1
                    parms[i] = tempparm
                end
            end
            return parms
        elseif ismmap_init&ismmap_run
            io = open(data[1]*".bin")
            dataset = Mmap.mmap(io,data[2],samplesize)
            # close(io)

            io2 = open(fname*"-run.bin", "w+")
            parmstype = Vector{SVector{parmdim,Float64}}
            parms = Mmap.mmap(io2, parmstype, numsteps÷keepperiod)
            parms[1]= initparm
            i = 1
            tempparm = initparm
            Mmap.sync!(parms)

            @showprogress for iternum in 2:numsteps
                batchdata = dataset[sample(1:samplesize, batchsize; replace = replacement),:]
                tempparm = sgld_update_1step(tempparm, batchdata, gradloglik, gradlogprior,
                    samplesize, batchsize, parmdim;
                    stepsize=stepsize/(1+iternum)^power, invtemp=invtemp, precon=precon, sqrtaniso=sqrtaniso)
                if (iternum % keepperiod == 1)
                    i += 1
                    parms[i] = tempparm
                end
                if (i % writeperiod ==0)
                    Mmap.sync!(parms)
                end
            end
            Mmap.sync!(parms)
            close(io)
            close(io2)
            dataset = nothing
            parms = nothing
            return (data[1]*"-"*fname, parmstype)
        elseif (!ismmap_init)&ismmap_run
            dataset = data
            io2 = open(fname*"-run.bin", "w+")
            parmstype = Vector{SVector{parmdim,Float64}}
            parms = Mmap.mmap(io2, parmstype, numsteps ÷ keepperiod)
            parms[1]= initparm
            i = 1
            tempparm = initparm
            Mmap.sync!(parms)

            @showprogress for iternum in 2:numsteps
                batchdata = dataset[sample(1:samplesize, batchsize; replace = replacement),:]
                tempparm = sgld_update_1step(tempparm, batchdata, gradloglik, gradlogprior,
                    samplesize, batchsize, parmdim;
                    stepsize=stepsize/(1+iternum)^power, invtemp=invtemp, precon=precon, sqrtaniso=sqrtaniso)
                if (iternum % keepperiod == 1)
                    i += 1
                    parms[i] = tempparm
                end
                if (i % writeperiod ==0)
                    Mmap.sync!(parms)
                end
            end
            Mmap.sync!(parms)
            close(io2)
            dataset = nothing
            parms = nothing
            return (fname*"-run", parmstype)
        elseif (ismmap_init)&(!ismmap_run)
            io = open(data[1]*".bin")
            dataset = Mmap.mmap(io,data[2],samplesize)

            parms = Array{typeof(initparm),1}(undef, numsteps)
            parms[1]= initparm

            @showprogress for iternum in 2:numsteps
                batchdata = dataset[sample(1:samplesize, batchsize; replace = replacement),:]
                parms[iternum] = sgld_update_1step(parms[iternum-1], batchdata, gradloglik, gradlogprior,
                    samplesize, batchsize, parmdim;
                    stepsize=stepsize, invtemp=invtemp, precon=precon, sqrtaniso=sqrtaniso)
            end
            return parms
        else
            error("invalid ismmap parameters")
        end
    end
end

function sgldPreSpec(algo, initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
    parmdim, numsteps, Jinv, Vinv, sqrtJinv, w, ismmap_init, ismmap_run; keepperiod=1,fnameprefix="")
    if ismmap_run
        fname = fnameprefix*string(algo)
    else
        fname = nothing
    end

    if algo == :vanillaSGD
        sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
            parmdim, numsteps, ismmap_init, ismmap_run;
            stepsize=4*batchsize/samplesize^2, invtemp=false, precon=false, sqrtaniso=false, replacement=true, fname=fname, keepperiod=keepperiod)
    elseif algo == :vanillaSGLDdecreasing
        sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
            parmdim, numsteps, ismmap_init, ismmap_run;
            stepsize=batchsize/8/samplesize^2, invtemp=1, precon=false, sqrtaniso=false, replacement=true, fname=fname, power=(1/3), keepperiod=keepperiod)
    elseif algo == :preconSGD
        sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
            parmdim, numsteps, ismmap_init, ismmap_run;
            stepsize=4*batchsize/samplesize^2, invtemp=false, precon=Jinv, sqrtaniso=false, replacement=true, fname=fname, keepperiod=keepperiod)
    elseif algo == :preconVSGD
        sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
            parmdim, numsteps, ismmap_init, ismmap_run;
            stepsize=4*batchsize/samplesize^2, invtemp=false, precon=Vinv, sqrtaniso=false, replacement=true, fname=fname, keepperiod=keepperiod)
    elseif algo == :vanillaSGLD
        # sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
        #     parmdim, numsteps, ismmap;
        #     stepsize=4*batchsize*(1-w)/samplesize^2, invtemp=1/w, precon=false, sqrtaniso=false, replacement=true, fname=fname)
        sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
            parmdim, numsteps, ismmap_init, ismmap_run;
            stepsize=batchsize/samplesize^2, invtemp=1, precon=false, sqrtaniso=false, replacement=true, fname=fname, keepperiod=keepperiod)
        # sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
        #     parmdim, numsteps, ismmap_init, ismmap_run;
        #     stepsize=batchsize/samplesize^2, invtemp=1, precon=norm(Jinv), sqrtaniso=false, replacement=true, fname=fname, keepperiod=keepperiod)
    elseif algo == :preconSGLD
        sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
            parmdim, numsteps, ismmap_init, ismmap_run;
            stepsize=4*batchsize*(1-w)/samplesize^2, invtemp=1/w, precon=Jinv, sqrtaniso=sqrtJinv, replacement=true, fname=fname, keepperiod=keepperiod)
    elseif algo == :largeBatchVanillaSGLD
        sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
            parmdim, numsteps, ismmap_init, ismmap_run;
            stepsize=4/samplesize^2, invtemp=1, precon=false, sqrtaniso=false, replacement=true, fname=fname, keepperiod=keepperiod)
    elseif algo == :largeBatchPreconSGLD
        sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
            parmdim, numsteps, ismmap_init, ismmap_run;
            stepsize=4/samplesize^2, invtemp=1, precon=Jinv, sqrtaniso=sqrtJinv, replacement=true, fname=fname, keepperiod=keepperiod)
    ##### PATCHY POWERS
    elseif algo == :vanillaSGDPow
            sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
                parmdim, numsteps, ismmap_init, ismmap_run;
                stepsize=4*batchsize/samplesize^(1+2/3), invtemp=false, precon=false, sqrtaniso=false, replacement=true, fname=fname, keepperiod=keepperiod)
    elseif algo == :preconSGDPow
        sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
            parmdim, numsteps, ismmap_init, ismmap_run;
            stepsize=4*batchsize/samplesize^(1+2/3), invtemp=false, precon=Jinv, sqrtaniso=false, replacement=true, fname=fname, keepperiod=keepperiod)
    elseif algo == :vanillaSGLDPow
        sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
            parmdim, numsteps, ismmap_init, ismmap_run;
            stepsize=batchsize/samplesize^(1+2/3), invtemp=1, precon=false, sqrtaniso=false, replacement=true, fname=fname, keepperiod=keepperiod)
    elseif algo == :preconSGLDPow
        sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
            parmdim, numsteps, ismmap_init, ismmap_run;
            stepsize=4*batchsize*(1-w)/samplesize^(1+2/3), invtemp=1/w, precon=Jinv, sqrtaniso=sqrtJinv, replacement=true, fname=fname, keepperiod=keepperiod)
    #############################
    elseif algo == :preconSGDdiag
        sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
            parmdim, numsteps, ismmap_init, ismmap_run;
            stepsize=4*batchsize/samplesize^2, invtemp=false, precon=inv(Diagonal(inv(Jinv))), sqrtaniso=false, replacement=true, fname=fname, keepperiod=keepperiod)
    elseif algo == :preconVSGDdiag
        sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
            parmdim, numsteps, ismmap_init, ismmap_run;
            stepsize=4*batchsize/samplesize^2, invtemp=false, precon=inv(Diagonal(inv(Vinv))), sqrtaniso=false, replacement=true, fname=fname, keepperiod=keepperiod)
    elseif algo == :preconSGLDdiag
        sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
            parmdim, numsteps, ismmap_init, ismmap_run;
            stepsize=4*batchsize*(1-w)/samplesize^2, invtemp=1/w, precon=inv(Diagonal(inv(Jinv))), sqrtaniso=sqrt(inv(Diagonal(inv(Jinv)))), replacement=true, fname=fname, keepperiod=keepperiod)
    ##### POWERS FOR ITERAVG
    elseif algo == :vanillaSGDPowHalf
            sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
                parmdim, numsteps, ismmap_init, ismmap_run;
                stepsize=4*batchsize/samplesize^(1+1/2), invtemp=false, precon=false, sqrtaniso=false, replacement=true, fname=fname, keepperiod=keepperiod)
    elseif algo == :vanillaSGDPowThird
            sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
                parmdim, numsteps, ismmap_init, ismmap_run;
                stepsize=4*batchsize/samplesize^(1+1/3), invtemp=false, precon=false, sqrtaniso=false, replacement=true, fname=fname, keepperiod=keepperiod)
    elseif algo == :vanillaSGDPowTwoThird
            sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
                parmdim, numsteps, ismmap_init, ismmap_run;
                stepsize=4*batchsize/samplesize^(1+2/3), invtemp=false, precon=false, sqrtaniso=false, replacement=true, fname=fname, keepperiod=keepperiod)
    elseif algo == :vanillaSGDPowOne
            sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
                parmdim, numsteps, ismmap_init, ismmap_run;
                stepsize=4*batchsize/samplesize^(1+1), invtemp=false, precon=false, sqrtaniso=false, replacement=true, fname=fname, keepperiod=keepperiod)
    # Balance Batch and Step Size
    # c_b = batchsize/samplesize^(1-h)
    # stepsize= 4*c_b/samplesize^(1+h) = 4*batchsize/samplesize^(2)
    elseif algo == :vanillaSGDPowHalfSumOne
            sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
                parmdim, numsteps, ismmap_init, ismmap_run;
                stepsize=4*batchsize/samplesize^2, invtemp=false, precon=false, sqrtaniso=false, replacement=true, fname=fname, keepperiod=keepperiod)
    elseif algo == :vanillaSGDPowThirdSumOne
            sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
                parmdim, numsteps, ismmap_init, ismmap_run;
                stepsize=4*batchsize/samplesize^2, invtemp=false, precon=false, sqrtaniso=false, replacement=true, fname=fname, keepperiod=keepperiod)
    elseif algo == :vanillaSGDPowTwoThirdSumOne
            sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
                parmdim, numsteps, ismmap_init, ismmap_run;
                stepsize=4*batchsize/samplesize^2, invtemp=false, precon=false, sqrtaniso=false, replacement=true, fname=fname, keepperiod=keepperiod)
    elseif algo == :vanillaSGDPowOneSumOne
            sgld(initparm, data, gradloglik, gradlogprior, samplesize, batchsize,
                parmdim, numsteps, ismmap_init, ismmap_run;
                stepsize=4*batchsize/samplesize^2, invtemp=false, precon=false, sqrtaniso=false, replacement=true, fname=fname, keepperiod=keepperiod)
    else
        error("big OOF: the prespecified setting you asked for doesn't exist!")
    end
end

function sgldMultiPreSpec(algos, initparm, data, gradloglik, gradlogprior, samplesize, batchsizes,
    parmdim, numstepss, Jinv, Vinv, sqrtJinv, w, ismmap_init, ismmap_run; keepperiods=false, fnameprefix="")

    if (keepperiods==false)
        keepperiods = Dict(algo => 1 for algo in algos)
    end

    Dict(algo => sgldPreSpec(algo, initparm, data, gradloglik, gradlogprior, samplesize, batchsizes[algo],
        parmdim, numstepss[algo], Jinv, Vinv, sqrtJinv, w, ismmap_init, ismmap_run; keepperiod=keepperiods[algo],fnameprefix=fnameprefix) for algo in algos)
end

function sgldMultiPreSpecMT(algos, initparm, data, gradloglik, gradlogprior, samplesize, batchsizes,
    parmdim, numstepss, Jinv, Vinv, sqrtJinv, w, ismmap_init, ismmap_run; keepperiods=false, fnameprefix="")

    if (keepperiods==false)
        keepperiods = Dict(algo => 1 for algo in algos)
    end
    if (!ismmap_run)
        d = Dict{Symbol,Array{typeof(initparm),1}}()
    else
        d = Dict{Symbol,Tuple{String,DataType}}()
    end
    Threads.@threads for algo in algos
        d[algo] = sgldPreSpec(algo, initparm, data, gradloglik, gradlogprior, samplesize, batchsizes[algo],
            parmdim, numstepss[algo], Jinv, Vinv, sqrtJinv, w, ismmap_init, ismmap_run; keepperiod=keepperiods[algo], fnameprefix=fnameprefix)
    end
    d
end
