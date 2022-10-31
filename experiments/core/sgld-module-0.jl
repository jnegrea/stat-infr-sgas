# 
# module SGLDModule
#     source_path = @__DIR__

    ## Dependencies
    using Distributions
    using Plots
    using StatsPlots
    using KernelDensity
    using Statistics
    using StatsBase
    using Parameters
    using JLD2
    using LinearAlgebra
    using GLM
    using DataFrames
    using Mmap
    using StaticArrays
    using ProgressMeter
    using FFTW
    using RData
    using CSVFiles
    import CodecBzip2

    export Layers, layerer, 
        modelRun, modelPlot, modelSummary,
        gaussianLocationInit, linearRegressionInit, logisticRegressionInit, logisticRegressionInitHuge, airlineInit

    include(source_path*"/sgld-module-1-run.jl")

    include(source_path*"/sgld-module-2-plotutils.jl")

    include(source_path*"/sgld-module-3-simmodels.jl")

# end # End Module