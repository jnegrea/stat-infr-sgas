using JLD2
using StatsBase
using Statistics
using GLM
using DataFrames
using RData
using StaticArrays
using Mmap
using CSV

output_path = "../output/"
isdir(output_path) || mkdir(output_path)

source_data_folder = "../source-data/DataExpo2009-extracted/"

output_path_6 = output_path*"experiments-6-airline-data-full/"
isdir(output_path_6) || mkdir(output_path_6)

source_data_fname_prefix = source_data_folder*""

years = 2008:2008

function airlineDataFull_compile_sizes(fname, source_data_fname_prefix, years)
    print("determining columns to keep\n")
    source_data = DataFrame(CSV.File(source_data_folder*string(years[1])*".csv", missingstring="NA"))
    coltypes = eltype.(eachcol(source_data))
    dropcols = max.((coltypes .== Missing), (coltypes .== String), (coltypes .== Union{Missing,String})) 

    source_data = dropmissing(DataFrame(CSV.File(source_data_folder*string(years[1])*".csv", missingstring="NA", drop=dropcols)))
    coltypes = eltype.(eachcol(source_data))
    colnames = names(source_data)

    numrows = nrow(source_data)
    rowsyears=[(1,numrows) for year in years]
    
    for i in 2:length(years)
        print("counting rows of "*string(years[i])*"\n")
        numrows += nrow(dropmissing(DataFrame(CSV.File(source_data_folder*string(years[i])*".csv", missingstring="NA", drop=dropcols))))
        rowsyears[i] = (rowsyears[i-1][2]+1,numrows)
    end
    @save fname*"-sizes.jld2" dropcols coltypes colnames numrows rowsyears
end
airlineDataFull_compile_sizes(output_path_6*"airline-data-full", source_data_fname_prefix, years)


function airlineDataFull_compile(fname, source_data_fname_prefix, source_data_sizes_file, years; format=:VectorOfVectors)
    @load source_data_sizes_file dropcols coltypes colnames numrows rowsyears
    
    if format == :VectorOfVectors
        statictype = Vector{SVector{length(coltypes),Int16}}
        # statictype = Array{Int16,2}
            
        io = open(fname*".bin", "w+")
        mmap_source_data = Mmap.mmap(io, statictype, numrows)
        # mmap_source_data = Mmap.mmap(io, statictype, (numrows,length(coltypes)))
        
        for i in 1:length(years)
            print("writing year "*string(years[i])*" to mmap\n")
            source_data_year = dropmissing(DataFrame(CSV.File(source_data_folder*string(years[i])*".csv", missingstring="NA", drop=dropcols)))
            for j in 1:(rowsyears[i][2]-rowsyears[i][1]+1)
                mmap_source_data[rowsyears[i][1]+j-1] = tuple(source_data_year[j,:]...)
            end
            # mmap_source_data[rowsyears[i][1]:rowsyears[i][2],:] .= source_data_year
            Mmap.sync!(mmap_source_data)
        end
        close(io)
    elseif format == :Array
        # statictype = Vector{SVector{length(coltypes),Int16}}
        statictype = Array{Int16,2}
        
        rm(fname*".bin", force=true)    
        io = open(fname*".bin", "w+")
        # mmap_source_data = Mmap.mmap(io, statictype, numrows)
        mmap_source_data = Mmap.mmap(io, statictype, (numrows,length(coltypes)))
        
        for i in 1:length(years)
            print("writing year "*string(years[i])*" to mmap\n")
            source_data_year = dropmissing(DataFrame(CSV.File(source_data_folder*string(years[i])*".csv", missingstring="NA", drop=dropcols)))
            # for j in 1:(rowsyears[i][2]-rowsyears[i][1]+1)
            #     mmap_source_data[rowsyears[i][1]+j-1] = tuple(source_data_year[j,:]...)
            # end
            mmap_source_data[rowsyears[i][1]:rowsyears[i][2],:] .= source_data_year
            Mmap.sync!(mmap_source_data)
        end
        close(io)
    else 
        error("invalid format")
    end
    
    @save fname*"-meta.jld2" colnames coltypes
end
source_data_sizes_file = output_path_6*"airline-data-full"*"-sizes.jld2"
airlineDataFull_compile(output_path_6*"airline-data-full", source_data_fname_prefix, source_data_sizes_file, years)


function airlineDataFull_compile_df(fname, source_data_fname_prefix, source_data_sizes_file, years)
    @load source_data_sizes_file dropcols coltypes colnames numrows rowsyears
    
    source_data_df = Int16.(dropmissing(DataFrame(CSV.File(source_data_folder*string(years[1])*".csv", missingstring="NA", drop=dropcols))))
    
    for i in 2:length(years)
        print("appending year "*string(years[i])*" to df \n")
        append!(source_data_df, Int16.(dropmissing(DataFrame(CSV.File(source_data_folder*string(years[i])*".csv", missingstring="NA", drop=dropcols)))))
    end
    print("saving...")    
    @save fname*"-df.jld2" source_data_df
end
source_data_sizes_file = output_path_6*"airline-data-full"*"-sizes.jld2"
airlineDataFull_compile_df(output_path_6*"airline-data-full", source_data_fname_prefix, source_data_sizes_file, years)


function airlineTrueparm(fname, source_data_file, source_data_metadata_file, source_data_df_file, model_formula, model_gll, model_hess, parmdim)
    @load source_data_df_file source_data_df
    
    N = (size(source_data_df_file)[1])
    shuffle_indices=sample(1:N,N; replace=false, ordered=false)
    source_data_df_file = source_df[shuffle_indices,:]
    shuffle_indices = nothing
    GC.gc()
    trueparm = coef(glm(model_formula, source_data_df_file, Binomial()))
    source_data_df_file = nothing
    GC.gc()
    
    statictype = Vector{SVector{length(coltypes),Int16}}
    io = open(fname*".bin")
        source_data = Mmap.mmap(io, statictype)

        J = zeros(parmdim,parmdim) 
        V = zeros(parmdim,parmdim)  
        for i in 1:N
            J += model_hess(trueparm,source_data[i,:])
            V += model_gll(trueparm,source_data[i,:])*model_gll(source_data[i,:],trueparm)'
        end
        J /= N
        V /= N
        Jinv = inv(J)
        sqrtJinv = sqrt(sqrt(Jinv*Jinv'))
        GC.gc()
    close(io)
    
    @save fname*".jld2" parmdim Jinv V sqrtJinv trueparm model_gll
end

# airlineTrueparm(output_path_4*"airline-data-trueparm",source_data_file)
