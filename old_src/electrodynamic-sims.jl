using StaticArrays
using Parameters
using OrdinaryDiffEq
using LaserTypes
using LinearAlgebra
using Statistics
using ThreadsX
using Plots
using LaTeXStrings
using Distributions
using RecursiveArrayTools
using Serialization

function folder_setup(path)
    path = replace(path,"/" => "")

    if split(pwd(),"/")[end] == path
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

cd(@__DIR__)
folder_setup("output5")

const N = 28_750 # number of particles, for each ring, 4N for radius averaging
const c = 137.036 # speed of light
const ω = 0.057; const T₀ = 2π/ω # angular frequency, period
const k = ω/c # wavenumber
const λ = c*T₀ # wavelength
const w₀ = 5 * λ # is the waist radius
const a₀ = 2.; const AA = a₀*c # (qA₀)/(mc) = a₀
const w = 10.; const τ = w/ω; # temporal pulse decay time given in terms of number of oscilations and frequency 
const n = 5 # number of periods to integrate before and after pulse collides with particles 
const π₀ = a₀*c # in atomic units, a₀mc has units of linear momentum and sets the scale for linear momentum transfered to the particle, not that m = 1 for our particles
const maxR = 3.25w₀ # maximum radius for distributing praticles

const g = @SMatrix [1  0  0  0;
                    0 -1  0  0;
                    0  0 -1  0;
                    0  0  0 -1] # Minkowski space metric

@with_kw struct InterPars{qq,mm,typo,rewts}
    q::qq = -1
    m::mm = 1
    type::typo
    roots::rewts = Float64[]
end


################################################################################
#Right Circularly Polarized

# p = 0
CPLGTYPE00 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = 0, p = 0,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))
CPLGTYPE0_1 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = -1, p = 0,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))
CPLGTYPE01 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = 1, p = 0,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))

# p = 1
CPLGTYPE10 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = 0, p = 1,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))
CPLGTYPE1_1 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = -1, p = 1,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))
CPLGTYPE11 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = 1, p = 1,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))

# p = 2
CPLGTYPE2_1 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = -1, p = 2,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))
CPLGTYPE21 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = 1, p = 2,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))
CPLGTYPE20 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = 0, p = 2,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.)) 
CPLGTYPE2_2 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = -2, p = 2,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))
CPLGTYPE22 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = 2, p = 2,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))


################################################################################
#Left Circularly Polarized

# p = 0
MCPLGTYPE00 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = 0, p = 0,  ξx = (1. + 0im)/√2, ξy = (-1im)/√2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))
MCPLGTYPE0_1 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = -1, p = 0,  ξx = (1. + 0im)/√2, ξy = (-1im)/√2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))
MCPLGTYPE01 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = 1, p = 0,  ξx = (1. + 0im)/√2, ξy = (-1im)/√2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))

# p = 1
MCPLGTYPE10 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = 0, p = 1,  ξx = (1. + 0im)/√2, ξy = (-1im)/√2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))
MCPLGTYPE1_1 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = -1, p = 1,  ξx = (1. + 0im)/√2, ξy = (-1im)/√2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))
MCPLGTYPE11 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = 1, p = 1,  ξx = (1. + 0im)/√2, ξy = (-1im)/√2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))


################################################################################
#Linearly Polarized

# p = 0
LPLGTYPE00 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = 0, p = 0, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))
LPLGTYPE0_1 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = -1, p = 0, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))
LPLGTYPE01 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = 1, p = 0, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))

# p = 1
LPLGTYPE10 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = 0, p = 1, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))
LPLGTYPE1_1 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = -1, p = 1, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))
LPLGTYPE11 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = 1, p = 1, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))

# p = 2
LPLGTYPE20 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = 0, p = 2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))
LPLGTYPE21 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = 1, p = 2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))
LPLGTYPE22 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = 2, p = 2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))
LPLGTYPE2_1 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = -1, p = 2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))
LPLGTYPE2_2 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, ϕ₀ = -π/2,
    w₀ = w₀, m = -2, p = 2, profile = GaussProfile(c = c, τ = w/ω, z₀ = 0.))



field_param = (
    CPLGTYPE0_1 = InterPars(type = CPLGTYPE0_1, roots = [0., 1.954]),
    CPLGTYPE00 = InterPars(type = CPLGTYPE00, roots = [0., 1.518]),
    CPLGTYPE01 = InterPars(type = CPLGTYPE01, roots = [0., 1.954]),
    CPLGTYPE1_1 = InterPars(type = CPLGTYPE1_1, roots = [0., 1., 2.566]),
    CPLGTYPE10 = InterPars(type = CPLGTYPE10, roots = [ 0., 0.707, 2.323]),
    CPLGTYPE11 = InterPars(type = CPLGTYPE11, roots = [0., 1., 2.566]),
    CPLGTYPE2_2 = InterPars(type = CPLGTYPE2_2, roots = [0., 1.000, 1.731, 3.176]),
    CPLGTYPE2_1 = InterPars(type = CPLGTYPE2_1, roots = [0., 0.797, 1.538, 3.005]),
    CPLGTYPE20 = InterPars(type = CPLGTYPE20, roots = [0., 0.541, 1.306, 2.807]),
    CPLGTYPE21 = InterPars(type = CPLGTYPE21, roots = [0., 0.797, 1.538, 3.005]),
    CPLGTYPE22 = InterPars(type = CPLGTYPE22, roots = [0., 1.000, 1.731, 3.176]),
    MCPLGTYPE0_1 = InterPars(type = MCPLGTYPE0_1, roots = [0., 1.954]),
    MCPLGTYPE00 = InterPars(type = MCPLGTYPE00, roots = [0., 1.518]),
    MCPLGTYPE01 = InterPars(type = MCPLGTYPE01, roots = [0., 1.954]),
    MCPLGTYPE1_1 = InterPars(type = MCPLGTYPE1_1, roots = [0., 1., 2.566]),
    MCPLGTYPE10 = InterPars(type = MCPLGTYPE10, roots = [0., 0.707, 2.323]),
    MCPLGTYPE11 = InterPars(type = MCPLGTYPE11, roots = [0., 1., 2.566]),
    LPLGTYPE0_1 = InterPars(type = LPLGTYPE0_1, roots = [0., 1.954]),
    LPLGTYPE00 = InterPars(type = LPLGTYPE00, roots = [0., 1.518]),
    LPLGTYPE01 = InterPars(type = LPLGTYPE01, roots = [0., 1.954]),
    LPLGTYPE1_1 = InterPars(type = LPLGTYPE1_1, roots = [0., 1., 2.566]),
    LPLGTYPE10 = InterPars(type = LPLGTYPE10, roots = [0., 0.707, 2.323]),
    LPLGTYPE11 = InterPars(type = LPLGTYPE11, roots = [0., 1., 2.566]),
    LPLGTYPE2_2 = InterPars(type = LPLGTYPE2_2, roots = [0., 1.000, 1.731, 3.176]),
    LPLGTYPE2_1 = InterPars(type = LPLGTYPE2_1, roots = [0., 0.797, 1.538, 3.005]),
    LPLGTYPE20 = InterPars(type = LPLGTYPE20, roots = [0., 0.541, 1.306, 2.807]),
    LPLGTYPE21 = InterPars(type = LPLGTYPE21, roots = [0., 0.797, 1.538, 3.005]),
    LPLGTYPE22 = InterPars(type = LPLGTYPE22, roots = [0., 1.000, 1.731, 3.176])
)

function dynEq(v,param,τ)
    @unpack q, m, type = param
    x = v[1:4]
    u = v[5:8]

    dx = u
    du = q/m * Fμν(x, type) * g * u

    return vcat(dx,du)
end

function Mμν(v)
    x⁰, x¹, x², x³, u⁰, u¹, u², u³ = v
    Lx = x²*u³ - x³*u²
    Ly = x³*u¹ - x¹*u³
    Lz = x¹*u² - x²*u¹
    # Nx = x¹*u⁰ - x⁰*u¹
    # Ny = x²*u⁰ - x⁰*u²
    # Nz = x³*u⁰ - x⁰*u³
    L² = Lx^2 + Ly^2 + Lz^2
    # N² = Nx^2 + Ny^2 + Nz^2
    #the physical quantities are m* [1,2,3] , m^2*[4], m/c* [5,6,7] , m^2/c^2*[8],
    return (Lx = Lx, Ly = Ly, Lz = Lz, L² = L², #Nx = Nx, Ny = Ny, Nz = Nz, N² = N²,
            x⁰ = x⁰, x¹ = x¹, x² = x², x³ = x³, u⁰ = u⁰, u¹ = u¹, u² = u², u³ = u³)
end

function σA(A,A²)
    .√abs.((A²-A.^2))
end

function Mμντ(X)
    [Mμν(x) for x in X.u]
end

function UniformAnnulus(r₁,r₂)
    θ = rand(Uniform(0,2*π))
    u = rand(Uniform(r₁,r₂))
    v = rand(Uniform(0,r₁+r₂))
    ρ = v<u ? u : r₁+r₂ - u
    [SVector{3}(ρ*cos(θ), ρ*sin(θ),0), ρ]
end

RHO = [UniformAnnulus(0,maxR) for i in 1:4N] 
RHOVec = VectorOfArray(RHO)
idx = sortperm(RHO, lt=(x,y)->x[2]<y[2])
R1 = RHOVec[1,:][idx]

τi = - n * τ
τf =   n * τ

const time_samples = 2000
const Δτ = range(0, stop = 2*n*τ/T₀, length = time_samples+1)

function xu_cat(X,U,T)
    x0 = c*T
    x = vcat([x0],X)
    u0 = √(c*c + U⋅U)
    u = vcat([u0],U)
    SVector{8}(vcat(x,u))
end
        
function problem_set(R0,p,U0 = [0,0,0])
    V0 = [xu_cat(r0,U0,τi) for r0 in R0]
    prob = ODEProblem(dynEq, V0[1], (τi, τf), p)
    function Ret_ODEProb(prob,i,repeat)
        ODEProblem(dynEq, V0[i], (τi, τf), p) #return
    end
    function output_func(sol,i)
        (Mμντ(sol), false) # put sol as argument, if you want the actual trajectories
    end
    EnsembleProblem(prob,prob_func = Ret_ODEProb, output_func = output_func)
end

function Particle_Avg(M,J)
    [ThreadsX.sum(getproperty(M[i][t], J)
    for i in 1:N)/N for t in 1:time_samples+1]
end

function Particle_Avg2(M,J)
    [ThreadsX.sum(getproperty(M[i][t], J)^2
    for i in 1:N)/N for t in 1:time_samples+1]
end

function Particle_AvgR(M)
    [ThreadsX.sum(hypot(getproperty(M[i][t], :x¹), getproperty(M[i][t], :x²))
    for i in 1:N)/N for t in 1:time_samples+1]
end

function Particle_AvgR²(M)
    [ThreadsX.sum(getproperty(M[i][t], :x¹)^2 + getproperty(M[i][t], :x²)^2
    for i in 1:N)/N for t in 1:time_samples+1]
end

# function replot!(plt, p)
#     plot!(plt, [], [], label="\$a_0=$a₀\$", color=nothing)
#     plot!(plt, [], [], label="\$p=$(p.p)\$", color=nothing)
#     plot!(plt, [], [], label="\$m=$(p.m)\$", color=nothing)
# end

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
circ = Shape(Plots.partialcircle(0, 2π))

function add_info!(plt, p, isscatter = false)
    #there is either a bug or confusing behaviour with ylims and aspect_ratio
    xl, yl = (isscatter ? ((-5.658368025329326, 5.667117711980884), (-3.662004592945377, 3.613120699860083)) : (xlims(plt), ylims(plt)))
    frame_width = xl[2] - xl[1]
    frame_height =yl[2] - yl[1]
    box_ratio = (w = 0.14 + 0.05 * abs(p.type.ξy*√2), h = 0.31) # hardcoded, I don't know how to get display length

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
    sgn = (imag(p.type.ξy) < 0 ? "-" : "")
    xi = (p.type.ξy == 0 ? "\$\\xi_y=0\$" : "\$\\xi_y= $sgn\\mathrm{i}/\\sqrt{2}\$") 
    
    annotate!(plt, [(text_x, text_y - 1δy, Plots.text("\$a_0=$a₀\$",       :center, 10, "computer modern")),
                    (text_x, text_y - 3δy, Plots.text(xi,                  :center, 10, "computer modern")),
                    (text_x, text_y - 5δy, Plots.text("\$p=$(p.type.p)\$", :center, 10, "computer modern")),
                    (text_x, text_y - 7δy, Plots.text("\$m=$(p.type.m)\$", :center, 10, "computer modern"))])
    
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

common_settings = ( fontfamily = "computer modern", 
                    dpi = 360, 
                    label = false, 
                    fillalpha=.3, 
                    color_palette = :Set1_4,
                    linewidth = 2)

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
Number of Oscillations in the Pulse,    w = $w 
Envelope Decay Time,                    τ = $τ  
Maximum Linear Momentum Transfered,     π₀ = $π₀ 
Disc Radius for Particle Distribution,  Rₘₐₓ = $maxR\n
""") # save out important parameters

for (sim_name, p) in pairs(field_param)
    println("$sim_name")
    folder_setup("$sim_name") # change folder for each simulation, as there are many files
    folder_setup("particles")
    write(io, "$sim_name\n")

    plot_data = [plot([],[], label = false) for i in 1:7]
    
    for j in 1:(length(p.roots) - 1)

        R0 = [UniformAnnulus(w₀*p.roots[j], w₀*p.roots[j+1])[1] for i in 1:N]
    
        eprob = problem_set(R0, p)
        M = solve(eprob, Vern9(), EnsembleThreads(), abstol=1e-9, reltol=1e-9,
                saveat = (τf-τi)/time_samples, trajectories = N)
        
        z,  Lz,  L² = Particle_Avg.((M,),(:x³, :Lz, :L²))
        z², Lz², L⁴ = Particle_Avg2.((M,),(:x³, :Lz, :L²))
    
        ρ  = Particle_AvgR(M)
        ρ² = Particle_AvgR²(M) 

        σz  = σA(z, z²)/w₀
        # σρ  = σA(ρ, ρ²)/w₀ # not worth computing because it's too wide and breaks plots
        σLz = σA(Lz, Lz²)/w₀/π₀
        σL² = σA(L², L⁴)/w₀/π₀/w₀/π₀

        write(io, 
            "from $(p.roots[j]/w₀) to $(p.roots[j+1]/w₀)\n"*
            "z ± σz $(z[end]/w₀) ± $(σz[end])\n"*
            "Lz ± σLz $(Lz[end]/w₀/π₀) ± $(σLz[end])\n"*
            "L² ± σL² $(L²[end]/w₀/π₀/w₀/π₀) ± $(σL²[end])\n"
             )

        plot_data[1] = plot!(plot_data[1], Δτ, z/w₀; 
                                common_settings...,
                                ribbon = σz, 
                                seriescolor = j, # these cannot be moved outside due to variable scope
                                xlabel = L"\tau/T_0", 
                                ylabel = L"\overline{z}/w_0")

        mean_ρ = 2/3*(p.roots[j+1]^3 - p.roots[j]^3)/(p.roots[j+1]^2 - p.roots[j]^2)

        plot_data[2] = plot!(plot_data[2], Δτ, ρ/w₀ .- mean_ρ; 
                                common_settings..., 
                                # ribbon = σρ, too big relative to ρ changes, makes plot unreadable               
                                seriescolor = j,
                                xlabel = L"\tau/T_0",
                                ylabel = L"\Delta\overline{\rho}/w_0")

        plot_data[3] = plot!(plot_data[3], Δτ, Lz/w₀/π₀;
                                common_settings..., 
                                ribbon = σLz,
                                seriescolor = j,
                                xlabel = L"\tau/T_0",
                                ylabel = L"\overline{L_z}/(w_0 \pi_0)")

        plot_data[4] = plot!(plot_data[4], Δτ, L²/w₀/π₀/w₀/π₀;
                                common_settings...,
                                ribbon = (L²/w₀/π₀/w₀/π₀, σL²),   
                                seriescolor = j,
                                xlabel = L"\tau/T_0",
                                ylabel = L"\overline{L^2}/(w_0^2 \pi_0^2)")

    end
    write(io,"\n")

    add_info!(plot_data[1], p)
    savefig(plot_data[1], "z_t_$(sim_name).png")
    add_info!(plot_data[2], p)
    savefig(plot_data[2], "rho_t_$(sim_name).png")
    add_info!(plot_data[3], p)
    savefig(plot_data[3], "Lz_t_$(sim_name).png")
    add_info!(plot_data[4], p)
    savefig(plot_data[4], "L2_t_$(sim_name).png")

    eprob = problem_set(R1, p)
    M = solve(eprob, Vern9(), EnsembleThreads(), abstol = 1e-9, reltol = 1e-9,
            saveat = (τf-τi)/time_samples, trajectories = 4N)

    X = [M[i][begin].x¹ for i in 1:4N]/w₀ # pulling out the initial x coordinates
    Y = [M[i][begin].x² for i in 1:4N]/w₀ # pulling out the initial y coordinates    
    Lz = [M[i][end].Lz for i in 1:4N] # pulling out the final Lz for colouring   
    Lzmax = max(abs.(extrema(Lz))...) # finding maximum
    colours = Lz/Lzmax # doing a normalization
    
    plot_data[7] = scatter(X,Y;
        marker_z = colours,
        markershape = circ,
        markerstrokealpha = 0,
        markersize = 1.2,
        seriescolor = :PuOr_9,
        clims = (-1., 1.),
        fontfamily = "computer modern", 
        dpi = 360, 
        label = false,  
        xlabel = L"x/w_0", ylabel = L"y/w_0", 
        aspect_ratio = 1,
        xticks = -4:1:4)
    add_info!(plot_data[7], p, true)
    savefig(plot_data[7], "XYLz.png")

    xDens = 100
    LzStats = zeros(xDens, 3)
    ρStats  = zeros(xDens, 3) 
    Midx = 1
    for i in 1:xDens
        ρ_avg = 0.  
        Lz_   = 0.
        Lz²_  = 0.
        ρ_    = 0.
        ρ²_   = 0.
        tally = 0
        m = M[Midx]
        ρ = hypot(m[begin].x¹, m[begin].x²)
        while ρ < i * maxR/xDens
            ρ_avg += ρ  
            Lz_   += m[end].Lz
            Lz²_  += m[end].Lz^2
            ρ_    += hypot(m[end].x¹, m[end].x²)
            ρ²_   += m[end].x¹^2 + m[end].x²^2
            tally += 1
            if Midx < 4N
                Midx += 1
            else 
                break 
            end 
            m = M[Midx]
            ρ = hypot(m[begin].x¹, m[begin].x²)    
        end
        LzStats[i,:] .= ρ_avg/tally, Lz_/tally, Lz²_/tally
        ρStats[i,:] .= ρ_avg/tally, ρ_/tally,  ρ²_/tally
    end  

    plot_data[5] = plot(LzStats[:,1]/w₀, LzStats[:,2]/π₀/w₀, 
                        label = false,
                        ribbon = σA(LzStats[:,2],LzStats[:,3])/π₀/w₀)
    plot_data[5] = add_rect!(plot_data[5], LzStats[:,1]/w₀, LzStats[:,2]/π₀/w₀, p.roots;
                        common_settings...,
                        ribbon = σA(LzStats[:,2],LzStats[:,3])/π₀/w₀,
                        seriescolor = 4,
                        xlabel = L"\rho_0/w_0", 
                        ylabel = L"\overline{L_z}(\rho_0)/(w_0 \pi_0)",
                        )
    add_info!(plot_data[5], p)
    savefig(plot_data[5], "Lz_rho_$(sim_name).png")
    
    plot_data[6] = plot(ρStats[:,1]/w₀, ρStats[:,2]/w₀ .- ρStats[:,1]/w₀;
                        label = false,
                        ribbon = σA(ρStats[:,2],ρStats[:,3])/w₀)
    plot_data[6] = add_rect!(plot_data[6], ρStats[:,1]/w₀, ρStats[:,2]/w₀ .- ρStats[:,1]/w₀, p.roots;
                        common_settings...,
                        ribbon = σA(ρStats[:,2],ρStats[:,3])/w₀,
                        seriescolor = 4,
                        xlabel = L"\rho_0/w_0", 
                        ylabel = L"\Delta\overline{\rho}_{\mathrm{final}}(\rho_0)/w_0")
    add_info!(plot_data[6], p)
    savefig(plot_data[6], "rho_rho_$(sim_name).png")

    serialize("$(sim_name)", plot_data)
    cd("..")
    cd("..")
    println("done")
end

close(io)
cd("..")

using Telegram
using Telegram.API: sendAnimation, sendMessage, sendSticker
using ConfigEnv
dotenv("../.env")

sendAnimation(animation = "CgACAgQAAxkBAAMWYI1V67io3ILX4lFH-PzAZmoGfugAAi0CAAIvno1SOEmh8Ay-BCYfBA", caption = "Hei, tu!", disable_notification = true)
sendMessage(text = "Fișierul `$(basename(@__FILE__))`, de pe $(ENV["HOSTNAME"]) a fost executat cu succes\\! 💯", parse_mode = "MarkdownV2")
sendSticker(sticker ="CAACAgIAAxkBAAMXYI1V9sq0HUcFb7jdWxkWC5bboqoAAg4AA-nYEygTpj1DX_hIHx8E", caption = "Totul a mers ok!", disable_notification = true)
