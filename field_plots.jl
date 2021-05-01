using StaticArrays
using Parameters
using LaserTypes
using LinearAlgebra
using Statistics
using ThreadsX
using Plots
using LaTeXStrings
using RecursiveArrayTools
using QuadGK
using Serialization

function folder_setup(path)
    path = replace(path,"/" => "")

    if split(pwd(),"/")[end] == path
        println("folder ok")
        return
    end

    if !isdir(path)
        mkdir(path)
        println("making folder")
    end

    cd(path)
    pwd()
end

folder_setup("output_fields_final2")

const c = 137.036
const ω = 0.057; const T₀ = 2π/ω
const k = ω*c
const λ = c*T₀
const w₀ = 75 * λ
const a₀ = 2.;const AA = a₀*c # (qA₀)/(mc) = a₀
const w = 10.; const τ = w/ω;
const n = 5
const π₀ = a₀*c
const maxR = 3w₀

@with_kw struct InterPars{qq,mm,typo,rewts}
    q::qq = -1
    m::mm = 1
    type::typo
    roots::rewts = Float64[]
end


