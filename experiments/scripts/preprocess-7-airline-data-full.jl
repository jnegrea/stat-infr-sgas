using JLD2
using StatsBase
using Statistics
#using GLM
using DataFrames
using RData
using StaticArrays
using Mmap
using CSV
using CategoricalArrays

output_path = "../output/"
isdir(output_path) || mkdir(output_path)

source_data_folder = "../source-data/DataExpo2009/"

output_path_7 = output_path*"experiments-7-airline-data-full/"
isdir(output_path_7) || mkdir(output_path_7)

source_data_fname_prefix = source_data_folder*""
include_str = true
# include_str = false
years = 2008:2008

function airlineDataFull_compile_sizes(fname, source_data_fname_prefix, years; include_str = false)
    print("determining columns to keep\n")
    source_data = DataFrame(
        CSV.File(
            source_data_folder * string(years[1]) * ".csv",
            missingstring = "NA",
        ),
    )
    coltypes = eltype.(eachcol(source_data))
    ##println the string columns' names
    stringcols = max.((coltypes .== Missing), (coltypes .== String3), (coltypes .== String7), (coltypes .== Union{Missing,String3}),(coltypes .== Union{Missing,String3}))
    ind = findall(>(0), stringcols)
    colnames = names(source_data)
    stringcols_names = colnames[ind]
    println("The string columns are ", stringcols_names, "\n")

    if include_str
        dropcols = max.((coltypes .== Missing))
    else
        dropcols = stringcols
    end
    println("columns to drop ", dropcols)
    #stringcols =max.( (coltypes .== String), (coltypes .== Union{Missing,String}))
    println("The column types before transformation is ", coltypes)
    println("The dimensions of the original dataset is ", size(source_data))
    source_data = dropmissing(
        DataFrame(
            CSV.File(
                source_data_folder * string(years[1]) * ".csv",
                missingstring = "NA",
                drop = dropcols,
            ),
        ),
    )

    coltypes = eltype.(eachcol(source_data))
    colnames = names(source_data)
    println("The column names after transformation are ", colnames, "\n")
    println("The column types after transformation is ", coltypes, "\n")
    println("The dimensions of the trimmed dataset is ", size(source_data), "\n")

    numrows = nrow(source_data)
    rowsyears = [(1, numrows) for year in years]

    for i = 2:length(years)
        print("counting rows of " * string(years[i]) * "\n")
        numrows += nrow(
            dropmissing(
                DataFrame(
                    CSV.File(
                        source_data_folder * string(years[i]) * ".csv",
                        missingstring = "NA",
                        drop = dropcols,
                    ),
                ),
            ),
        )
        rowsyears[i] = (rowsyears[i-1][2] + 1, numrows)
    end
    #numrows: number of rows in the dataset
    #rowsyears: years by its rows tuple
    post_fix = "-sizes"
    if include_str
        post_fix *= "-all"
    end
    @save fname * post_fix * ".jld2" dropcols coltypes colnames numrows rowsyears include_str
end
fname = output_path_7*"airline-data-full"
airlineDataFull_compile_sizes(output_path_7*"airline-data-full", source_data_fname_prefix, years; include_str = include_str)

function airlineDataFull_compile_df(
    fname,
    source_data_fname_prefix,
    source_data_sizes_file,
    years,
)
    @load source_data_sizes_file dropcols coltypes colnames numrows rowsyears include_str

    source_data_df =
        #Int16.(
            dropmissing(
                DataFrame(
                    CSV.File(
                        source_data_folder * string(years[1]) * ".csv",
                        missingstring = "NA",
                        drop = dropcols,
                    ),
                ),
            )#,
        #)
    #
    for i = 2:length(years)
        print("appending year " * string(years[i]) * " to df \n")
        append!(
            source_data_df,
            #Int16.(
                dropmissing(
                    DataFrame(
                        CSV.File(
                            source_data_folder * string(years[i]) * ".csv",
                            missingstring = "NA",
                            drop = dropcols,
                        ),
                    ),
                ),
            #),
        )
    end
    if include_str
        transform!(source_data_df, names(source_data_df, Union{Missing,AbstractString}) .=> categorical, renamecols=false)
    end
    source_data_fname_postfix = "-df"
    if include_str
        source_data_fname_postfix *= "-all"
    end
    print("saving...")
    @save fname * source_data_fname_postfix*".jld2" source_data_df include_str
end

source_data_sizes_file = output_path_7 * "airline-data-full" * "-sizes"
if include_str
    source_data_sizes_file *=  "-all"
end
source_data_sizes_file *=  ".jld2"

airlineDataFull_compile_df(
    output_path_7 * "airline-data-full",
    source_data_fname_prefix,
    source_data_sizes_file,
    years,
)
