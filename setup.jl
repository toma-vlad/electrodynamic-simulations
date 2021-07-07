cd(@__DIR__)

# if isfile("Project.toml") && isfile("Manifest.toml")
#     using Pkg
#     Pkg.activate(".")
# else 
#     printstyled("You are missing environment files!\n"; color = :red, bold = true)
# end

using StaticArrays
using LaserTypes
using LinearAlgebra

using Plots
fmt = "pdf"

using LaTeXStrings
using RecursiveArrayTools
using Serialization
using OrdinaryDiffEq
using ThreadsX
using Distributions
using QuadGK
using UnPack
using LaserTypes: fundamental_constants

function folder_setup(path)
    path = replace(path,"/" => "")

    if basename(pwd()) == basename(path) # split(pwd(),"/")[end] == path
        println("already in $path")
        return
    end

    if !isdir(path)
        mkdir(path)
        println("making folder $path")
    end

    cd(path)
    println("moved in $path")
end

const N = 100_000 # number of particles
const c = 137.036 # speed of light
const ω = 0.057; const T₀ = 2π/ω # angular frequency, period
const k = ω/c # wavenumber
const λ = c*T₀ # wavelength
const w₀ = 75 * λ # is the waist radius
const a₀ = 1e-5; const AA = a₀*c # (qA₀)/(mc) = a₀
const w = 10.; const τ = w/ω; # temporal pulse decay time given in terms of number of oscilations and frequency 
const n = 5 # number of periods to integrate before and after pulse collides with particles 
const π₀ = a₀*c # in atomic units, a₀mc has units of linear momentum and sets the scale for linear momentum transfered to the particle, not that m = 1 for our particles
const maxR = 3.25w₀ # maximum radius for distributing praticles
const ϕ₀ = 0.
    
folder_setup("output_a$(a₀)_w$(round(w₀/λ, digits = 1))_phi$(round(ϕ₀, digits = 3))")

profile = GaussProfile(τ = w/ω, z₀ = 0.)

function init_lasers(λ, a₀, ϕ₀, w₀, p_max)
    Iterators.flatten([[setup_laser(LaguerreGaussLaser, :atomic; λ, a₀, ϕ₀, w₀, p, m, ξx, ξy, profile)
        for m in -p:p] 
        for p in 0:p_max, 
            (ξx, ξy) in [(1., 0.), 
                         ((1. + 0im)/√2, (1im)/√2), 
                         ((1. + 0im)/√2, (-1im)/√2)]     
    ])
end

# p_max = 2; 
lasers = init_lasers(λ, a₀, ϕ₀, w₀, 2)
lasers = [setup_laser(LaguerreGaussLaser, :atomic; λ, a₀, ϕ₀, w₀, p = 2, m = -2, ξx = 1., ξy = 0., profile)]

# roots don't change for m = ± |m|; adimensional ρ/w₀
rootsLUT = Base.ImmutableDict(
    (p = 0, m = 1) => [0., 1.954], # unphysical
    
    (p = 0, m = 0) => [0., 1.518],
    
    (p = 1, m = 0) => [0., 0.707, 2.323],
    (p = 1, m = 1) => [0., 1., 2.566],
    
    (p = 2, m = 0) => [0., 0.541, 1.306, 2.807],
    (p = 2, m = 1) => [0., 0.797, 1.538, 3.005],
    (p = 2, m = 2) => [0., 1.000, 1.731, 3.176]
)

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
circ = Shape(Plots.partialcircle(0, 2π))

function add_info!(plt, type, isscatter = false)
    #there is either a bug or confusing behaviour with ylims and aspect_ratio
    xl, yl = (isscatter ? ((-5.658368025329326, 5.667117711980884), (-3.662004592945377, 3.613120699860083)) : (xlims(plt), ylims(plt)))
    frame_width = xl[2] - xl[1]
    frame_height =yl[2] - yl[1]
    
    ξy = type.polarization.ξy
    
    box_ratio = (w = 0.14 + 0.05 * abs(ξy*√2), h = 0.31) # hardcoded, I don't know how to get display length

    # lower left corner coordinates
    box = (x = (0.99 - box_ratio.w)*frame_width + xl[1], 
           y = (0.99 - box_ratio.h)*frame_height + yl[1],
           w = box_ratio.w*frame_width,
           h = box_ratio.h*frame_height)

    # compute text positions 
    text_x = box.x + box.w/2 # always centered on the box 
    text_y = box.y + box.h # starting from the top, mind the minus
    δy = box.h/8
    
    # find the exact strings to print, related to polarization
    sgn = (imag(ξy) < 0 ? "-" : "")
    xi = (ξy == 0 ? "\$\\xi_y=0\$" : "\$\\xi_y= $sgn\\mathrm{i}/\\sqrt{2}\$") 
    
    annotate!(plt, [(text_x, text_y - 1δy, Plots.text("\$a_0=$(type.a₀)\$", :center, 10, "computer modern")),
                    (text_x, text_y - 3δy, Plots.text(xi,                   :center, 10, "computer modern")),
                    (text_x, text_y - 5δy, Plots.text("\$p=$(type.p)\$",    :center, 10, "computer modern")),
                    (text_x, text_y - 7δy, Plots.text("\$m=$(type.m)\$",    :center, 10, "computer modern"))])
    
    # add a rectangle without changing the frame size
    plot!(plt, rectangle(box.w, box.h, box.x, box.y); 
                        xlims = xl, ylims = yl,
                        linewidth = 1.5,
                        fillcolor = :white, label = false)
end

function add_rect!(plt, X, Y, roots; kwargs...) 
    h = ylims(plt)[2]-ylims(plt)[1]
    y₀ = ylims(plt)[1]
    underlay = plot([],[], label = false)
    for i in 1:(length(roots) - 1)
        a = roots[i]
        b = roots[i+1]
        plot!(underlay,rectangle(b-a, h, a, y₀), 
        color_palette = :Set1_4, seriescolor = i,
        fillalpha = .4, linecolor = nothing, label = false)
    end
    plot!(underlay, X, Y; 
        xlims = xlims(plt), ylims = ylims(plt), kwargs...) 
end

const τi = - n * τ
const τf =   n * τ
const time_samples = 2000
const Δτ = range(0, stop = 2*n*τ/T₀, length = time_samples+1)

function polarization(type)
    if imag(type.polarization.ξy)*√2 == 1.
        printstyled("\n Right Circular Polarization, p = $(type.p), m = $(type.m)\n"; color = :green, bold = true)
        return "RHCP"
    end
    if imag(type.polarization.ξy)*√2 == -1.
        printstyled("\n Left Circular Polarization, p = $(type.p), m = $(type.m)\n"; color = :green, bold = true)
        return "LHCP"
    end
    if imag(type.polarization.ξy)*√2 == 0.
        printstyled("\n Linear Polarization, p = $(type.p), m = $(type.m)\n"; color = :green, bold = true)
        return "LinP"
    end
end

io = open("output_log.txt","w")
write(io,"""Simulation Paramters

Number of Particles per Ring,           N = $N
Speed of Light [Atomic Units],          c = $c 
Angular Frequency [Atomic Units],       ω = $ω
Oscillation Period [Atomic Units],      T₀ = $T₀
Wavenumber [Atomic Units],              k = $k
Wavelength [Atomic Units],              λ = $λ 
Beam Waist Radius [Atomic Units],       w₀ = $w₀ 
Reduced Vector Potential,               a₀ = $a₀
Initial Phase                           φ₀ = $ϕ₀
Number of Oscillations in the Pulse,    w = $w 
Envelope Decay Time,                    τ = $τ  
Maximum Linear Momentum Transfered,     π₀ = $π₀ 
Disc Radius for Particle Distribution,  Rₘₐₓ = $maxR\n
""") # save out important parameters

# include("field-plots.jl")
include("electrodynamic-sims2.jl")

close(io)
cd("..") # return to file folder
