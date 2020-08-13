using StatsModels: StatsModels, AbstractTerm, FunctionTerm, term, terms, apply_schema, modelcols
using ShiftedArrays: lag
using DataFrames: DataFrame, groupby, transform!
using Underscores: @_

struct GroupedDiffTerm{T,G} <: AbstractTerm
    term::T
    nsteps::Int
    groups::G
end

groupeddiff(t, n, g) = GroupedDiffTerm(t, n, g)
groupeddiff(t::Symbol, n, g::Symbol) = GroupedDiffTerm(term(t), n, term(g))
groupeddiff(t, n, g::Symbol) = GroupedDiffTerm(t, n, term(g))

#GroupedDiffTerm(term, nsteps) = GroupedDiffTerm(term, nsteps, term(1))

StatsModels.terms(t::GroupedDiffTerm) = (t.term, t.groups)

function StatsModels.apply_schema(t::FunctionTerm{typeof(groupeddiff)},
                                  sch::StatsModels.Schema,
                                  Mod::Type)
    term_parsed, nsteps_parsed, groups_parsed = t.args_parsed
    nsteps = nsteps_parsed.n
    
    term = apply_schema(term_parsed, sch, Mod)
    groups = apply_schema(groups_parsed, sch, Mod)
    return GroupedDiffTerm(term, nsteps, groups)
end

# apply_schema to internal Terms and check for proper types
function StatsModels.apply_schema(t::GroupedDiffTerm,
                                  sch::StatsModels.Schema,
                                  Mod::Type)
    term = apply_schema(t.term, sch, Mod)
    groups = apply_schema(t.groups, sch, Mod)
    GroupedDiffTerm(term, t.nsteps, groups)
end

function StatsModels.modelcols(ft::GroupedDiffTerm, d::NamedTuple)
    column, groups = modelcols(terms(ft), d)
    df = DataFrame(column=column, groups=vec(groups))
    @_ groupby(df, :groups) |>
        transform!(__, :column => (x -> lead(x, ) .- lag(x, ft.nsteps)) => :column_diffed) |>
        __.column_diffed
end

StatsModels.width(ft::GroupedDiffTerm) = 1
StatsModels.coefnames(ft::GroupedDiffTerm) = "diff(" .* StatsModels.coefnames(ft.term) .* ", $(ft.nsteps))"
# # names variables from the data that a LogTerm relies on
StatsModels.termvars(p::GroupedDiffTerm) = StatsModels.termvars(p.term)
