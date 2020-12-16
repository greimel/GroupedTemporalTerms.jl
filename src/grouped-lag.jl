using StatsModels: StatsModels, AbstractTerm, FunctionTerm, term, terms, apply_schema, modelcols
using ShiftedArrays: lag
using DataFrames: DataFrame, groupby, transform
using Underscores: @_

struct GroupedLagTerm{T,G} <: AbstractTerm
    term::T
    nsteps::Int
    groups::G
end

groupedlag(t, n, g) = GroupedLagTerm(t, n, g)

GroupedLagTerm(term, nsteps) = GroupedLagTerm(term, nsteps, term(1))

StatsModels.terms(t::GroupedLagTerm) = (t.term, t.groups)

function StatsModels.apply_schema(t::FunctionTerm{typeof(groupedlag)},
                                  sch::StatsModels.Schema,
                                  Mod::Type)
    term_parsed, nsteps_parsed, groups_parsed = t.args_parsed
    nsteps = nsteps_parsed.n
    
    term = apply_schema(term_parsed, sch, Mod)
    groups = apply_schema(groups_parsed, sch, Mod)
    return GroupedLagTerm(term, nsteps, groups)
end

# apply_schema to internal Terms and check for proper types
function StatsModels.apply_schema(t::GroupedLagTerm,
                                  sch::StatsModels.Schema,
                                  Mod::Type)
    term = apply_schema(t.term, sch, Mod)
    groups = apply_schema(t.groups, sch, Mod)
    GroupedLagTerm(term, t.nsteps, groups)
end

function StatsModels.modelcols(ft::GroupedLagTerm, d::NamedTuple)
    column = modelcols(ft.term, d)
    group_var = StatsModels.termvars(ft.groups)[1]
    groups = d[group_var]
    df = DataFrame(column=column, groups=groups)
    @_ groupby(df, :groups) |>
        transform(__, :column => (x -> lag(x, ft.nsteps)) => :column_lagged) |>
        __.column_lagged
end

StatsModels.width(ft::GroupedLagTerm) = 1
StatsModels.coefnames(ft::GroupedLagTerm) = "lag(" .* StatsModels.coefnames(ft.term) .* ", $(ft.nsteps))"
# # names variables from the data that a LogTerm relies on
StatsModels.termvars(p::GroupedLagTerm) = StatsModels.termvars(p.term)
