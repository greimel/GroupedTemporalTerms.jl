using Pkg
pkg"add https://github.com/greimel/FixedEffectModels.jl#missings"

using GroupedTemporalTerms: groupedlag
using Test
using DataFrames 
using Underscores: @_
using FixedEffectModels: @formula, reg
using CategoricalArrays: categorical
using ShiftedArrays: lag
using StatsModels: StatsModels, schema, apply_schema, modelcols, coef

@testset "GroupedTemporalTerms.jl" begin
    # create fake data
    df = let
        group_ids = rand(1:3, 1000)
        groups = ["a", "b", "c"][group_ids]
        x = rand(1000)
        
        df = DataFrame(grp = groups, grp_id = group_ids, x = x)
        @_ groupby(df, :grp) |> 
            transform!(__, :x => lag => :x_lagged) |>
            transform!(__, [:grp_id, :x_lagged] => ByRow((g, xl) -> π * g + xl) => :y)
        
        df[!, :y] = coalesce.(df.y, 0.0)
        df
    end
    
    # check that correct model matrix is created
    f = @formula(y ~ grp + groupedlag(x, 1, grp))
    
    sch = schema(f, df)
    ff = apply_schema(f, sch)
    y, X = modelcols(ff, df)
    
    df_verify = DataFrame(X)
    rename!(df_verify, StatsModels.coefnames(ff)[2])
    
    @test isequal(df_verify."lag(x, 1)", df.x_lagged)

    # test regression with patched version of FixedEffectModels.jl
    out = reg(df, f)
    @test coef(out)[[1,end]] ≈ [π, 1]
end

