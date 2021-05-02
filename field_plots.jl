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

folder_setup("output_fields")

const c = 137.036 # speed of light
const ω = 0.057; const T₀ = 2π/ω # angular frequency, period
const k = ω*c # wavevector
const λ = c*T₀ # wavelength
const w₀ = 75 * λ # is the waist radius
const a₀ = 2.; const AA = a₀*c # (qA₀)/(mc) = a₀
const w = 10.; const τ = w/ω; # temporal pulse decay time given in terms of number of oscilations and frequency 
const n = 5 # number of periods to integrate before and after pulse collides with particles 
const π₀ = a₀*c # in atomic units, a₀mc has units of linear momentum and sets the scale for linear momentum transfered to the particle, not that m = 1 for our particles
const maxR = 4w₀ # maximum radius for distributing praticles

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

const τi = - n * τ
const τf =   n * τ
const time_samples = 2000
const Δτ = range(0, stop = 2*n*τ/T₀, length = time_samples+1)

const domainXY = (-maxR:λ:maxR)/w₀
const domainR  = (0.:λ:maxR)/w₀
const time = range(- n * τ, stop = n * τ, length = 800)/T₀

function replot!(plt, p)
    plot!(plt, [], [], label="\$a_0=$a₀\$", color=nothing)
    plot!(plt, [], [], label="\$p=$(p.p)\$", color=nothing)
    plot!(plt, [], [], label="\$m=$(p.m)\$", color=nothing)
end

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

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
        plot!(underlay, X, Y; kwargs...) 
end

function polarization(p)
    if imag(p.type.ξy)*√2 == 1.
        println("Right Circular Polarization, p = $(p.type.p), m = $(p.type.m)")
        "right-handedCP"
    end
    if imag(p.type.ξy)*√2 == -1.
        println("Left Circular Polarization, p = $(p.type.p), m = $(p.type.m)")
        "left-handedMCP"
    end
    if imag(p.type.ξy)*√2 == 0.
        println("Linear Polarization, p = $(p.type.p), m = $(p.type.m)")
        "linearLP"
    end
end

for p in field_param

    pol = polarization(p)

    # useful function definitions

    fieldE(x,y,z,t) = E([x,y,z],t,p.type)
    fieldB(x,y,z,t) = B([x,y,z],t,p.type)
    
    function wenergy(x,y,t) 
        w = fieldE(x,y,0,t)⋅fieldE(x,y,0,t) + c*c * fieldB(x,y,0,t)⋅fieldB(x,y,0,t)
        if isnan(w) || !isfinite(w)
            println("Your field definition may have issues.")
            0      
        else 
            w
        end
    end
    
    wenergy₀(x,y) = wenergy(x,y,0)
    wenergy_period(x,y) = quadgk(t -> wenergy(x,y,t), 0, T₀, rtol = 1e-8)[1]

    wρ(ρ) = quadgk(ϕ -> wenergy₀(ρ*cos(ϕ), ρ*sin(ϕ))/(2*π), 0, 2*π, rtol = 1e-8)[1]
    
    Eρ(ρ,i) = quadgk(ϕ -> fieldE(ρ*cos(ϕ), ρ*sin(ϕ),0,0)[i]/(2*π), 0, 2*π, rtol = 1e-8)[1]
    
    xyEx(x,y) = fieldE(x,y,0,0)[1]
    xyEz(x,y) = fieldE(x,y,0,0)[3]

    jz₀(x,y) = ([x,y,0.]×(fieldE(x,y,0,0)×fieldB(x,y,0,0)))[3]
    jzρ(ρ) = quadgk(ϕ -> jz₀(ρ*cos(ϕ), ρ*sin(ϕ))/(2*π), 0, 2*π, rtol = 1e-8, atol = 1.e-5)[1]

    # end of definitions
    
    # plots start here

    plot_data = [plot([],[], label = false) for i in 1:7]

    # 1. Heatmaps

    common_settings = (fontfamily = "computer modern", dpi = 360, label = false)
    plot_settings   = (color_palette = :Set1_4, seriescolor = 4, linewidth = 2.5)

    # deltaExz(x,y) = abs(fieldE(w₀*x,w₀*y,0,0)[1]) - abs(fieldE(w₀*x,w₀*y,0,0)[3])
    # dExz = heatmap(domainXY, domainXY, deltaExz, fontfamily = "computer modern",
    #             xlabel = L"\rho/w_0", ylabel = L"\rho/w_0", aspect_ratio = 1, dpi = 360)
    # replot!(dExz, p.type)
    # savefig(dExz, "dExz$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")
    # println("Polarization $(imag(p.type.ξy)*√2), p = $(p.type.p), m = $(p.type.m)")
    # println("dExz")
    
    plot_data[1] = heatmap(domainXY, domainXY, (x,y) -> wenergy₀(w₀*x, w₀*y);
                xlabel = L"\rho/w_0", ylabel = L"\rho/w_0", 
                color_palette = :lajolla,
                aspect_ratio = 1, common_settings...)
    replot!(plot_data[1], p.type) 
    savefig(plot_data[1], "xy00Energy$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")
    println("xyNRG₀")

    plot_data[2] = heatmap(domainXY, domainXY, (x,y) -> xyEx(w₀*x, w₀*y);
                xlabel = L"\rho/w_0", ylabel = L"\rho/w_0",
                color_palette = :vik, 
                aspect_ratio = 1, common_settings...)
    replot!(plot_data[2], p.type)
    savefig(plot_data[2], "xy00Ex$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")
    println("xy00Ex")

    plot_data[3] = heatmap(domainXY, domainXY, (x,y) -> xyEz(w₀*x, w₀*y);
                xlabel = L"\rho/w_0", ylabel = L"\rho/w_0", 
                color_palette = :vik, 
                aspect_ratio = 1, common_settings...)
    replot!(plot_data[3], p.type)
    savefig(plot_data[3], "xy00Ez$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")
    println("xy00Ez")

    # xyNRG = heatmap(domainXY, domainXY, wenergy_period, aspect_ratio = 1, dpi = 360)
    # replot!(xyNRG, p.type)
    # savefig(xyNRG, "xy00TEnergy$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")

    plot_data[4] = heatmap(domainXY, domainXY, (x,y) -> jz₀(w₀*x, w₀*y);
                xlabel = L"\rho/w_0", ylabel = L"\rho/w_0",
                color_palette = :vik, 
                aspect_ratio = 1, common_settings...)
    replot!(plot_data[4], p.type)
    savefig(plot_data[4], "xy00jz$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")
    println("xyjz₀")

    # 2. Azimuthally averaged radial plots

    jzρ_ = [jzρ(r*w₀) for r in domainR]
    plot_data[5] = plot(domainR, jzρ_, label = false)
    plot_data[5] = add_rect!(plot_data[5], domainR, jzρ_, p.roots;
                xlabel = L"\rho/w_0", ylabel = L"\overline{j_z^{xy}}(\rho, t = 0)\quad\mathrm{[a.u.]}", 
                common_settings..., plot_settings...)
    replot!(plot_data[5], p.type)
    savefig(plot_data[5], "rjz$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")
    println("jzρ")

    # Eρx = [Eρ(r*w₀,1) for r in domainR]
    # ρEρx = plot(domainR, Eρx, label = false, xlabel = L"\rho/w_0", dpi = 360)
    # replot!(ρEρx, p.type)
    # savefig(ρEρx, "rEx$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")
    # #println("Polarization $(imag(p.type.ξy)*√2), p = $(p.type.p), m = $(p.type.m)")
    # println("Eρx")

    # Eρz = [Eρ(r*w₀,3) for r in domainR]
    # ρEρz = plot(domainR, Eρz, label = false, xlabel = L"\rho/w_0", dpi = 360)
    # replot!(ρEρz, p.type)
    # savefig(ρEρz, "rEz$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")
    # #println("Polarization $(imag(p.type.ξy)*√2), p = $(p.type.p), m = $(p.type.m)")
    # println("Eρz")
    
    wρ_ = [wρ(r*w₀) for r in domainR]
    plot_data[6] = plot(domainR, wρ_, label = false)
    plot_data[6] = add_rect!(plot_data[6], domainR, wρ_, p.roots; 
        xlabel = L"\rho/w_0", ylabel = L"\mathrm{Intensity}(\rho)\quad\mathrm{[a.u.]}",
        common_settings..., plot_settings...)
    replot!(plot_data[6], p.type)
    savefig(plot_data[6], "box_rw$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")
    println("jzρ")

    # 3. Time axis plots.

    # tw_ = [wenergy(1/√2/2, 1/√2/2, ti*T₀) for ti in time]
    # tw = plot(time, tw_, label = false, xlabel = L"t/T_0", dpi = 360)
    # replot!(tw, p.type)
    # savefig(tw,"tw$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")
    # #println("Polarization $(imag(p.type.ξy)*√2), p = $(p.type.p), m = $(p.type.m)")
    # println("tw")

    tE_ = [fieldE(w₀/√2/2, w₀/√2/2, 0, ti*T₀)[1] for ti in time]
    plot_data[7] = plot(time, tE_;
         xlabel = L"t/T_0", ylabel = L"E_x({w_0}/{\sqrt{8}}, {w_0}/{\sqrt{8}}, 0, t)\quad\mathrm{[a.u.]}",
         common_settings..., plot_settings...)
    replot!(plot_data[7], p.type)
    savefig(plot_data[7],"tE$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")
    println("tE")

    serialize("plot_data_$(pol)_$(p.type.p)_$(p.type.m)", plot_data)
end
