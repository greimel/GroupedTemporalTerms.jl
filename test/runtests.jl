using Pkg
pkg"add https://github.com/greimel/FixedEffectModels.jl#missings"

using GroupedTemporalTerms: groupedlag
using Test
using DataFrames, Underscores
using FixedEffectModels#: @formula, reg
using ShiftedArrays: lag
using StatsModels#: StatsModels, schema, apply_schema, modelcols



@testset "GroupedTemporalTerms.jl" begin
    df = DataFrame(y = ones(10),
                   x =   [1,   2  , 11,  12,  3,   13,  4,   5,   14,  15],
                   grp = ["a", "a", "b", "b", "a", "b", "a", "a", "b", "b" ]
                   )

    f = @formula(y ~ x + grp + groupedlag(x, 1, grp))
    
    sch = schema(f, df)
    ff = apply_schema(f, sch)
    y, X = modelcols(ff, df)
    
    df_verify = DataFrame(X)
    rename!(df_verify, [:x, :grp, :x_lagged])
    df_verify[!,:grp] = copy(df.grp)
    
    @_ groupby(df_verify, :grp) |> 
        transform!(__, :x => lag => :x_lagged_verify)
        
    @test isequal(df_verify.x_lagged, df_verify.x_lagged_verify)
    
        
    out = reg(df, f)
    @test coef(out) ≈ [0, 1, 0, -1]
end

