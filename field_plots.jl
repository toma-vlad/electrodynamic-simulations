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
const a₀ = 8.; const AA = a₀*c # (qA₀)/(mc) = a₀
const w = 10.; const τ = w/ω; # temporal pulse decay time given in terms of number of oscilations and frequency 
const n = 5 # number of periods to integrate before and after pulse collides with particles 
const π₀ = a₀*c # in atomic units, a₀mc has units of linear momentum and sets the scale for linear momentum transfered to the particle, not that m = 1 for our particles
const maxR = 3.25w₀ # maximum radius for distributing praticles

@with_kw struct InterPars{qq,mm,typo,rewts}
    q::qq = -1
    m::mm = 1
    type::typo
    roots::rewts = Float64[]
end


################################################################################
#Right Circularly Polarized

# p = 0
CPLGTYPE00 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = 0, p = 0,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))
CPLGTYPE0_1 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = -1, p = 0,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))
CPLGTYPE01 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = 1, p = 0,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))

# p = 1
CPLGTYPE10 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = 0, p = 1,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))
CPLGTYPE1_1 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = -1, p = 1,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))
CPLGTYPE11 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = 1, p = 1,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))

# p = 2
CPLGTYPE2_1 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = -1, p = 2,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))
CPLGTYPE21 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = 1, p = 2,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))
CPLGTYPE20 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = 0, p = 2,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.)) 
CPLGTYPE2_2 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = -2, p = 2,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))
CPLGTYPE22 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = 2, p = 2,  ξx = (1. + 0im)/√2, ξy = (1im)/√2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))


################################################################################
#Left Circularly Polarized

# p = 0
MCPLGTYPE00 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = 0, p = 0,  ξx = (1. + 0im)/√2, ξy = (-1im)/√2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))
MCPLGTYPE0_1 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = -1, p = 0,  ξx = (1. + 0im)/√2, ξy = (-1im)/√2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))
MCPLGTYPE01 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = 1, p = 0,  ξx = (1. + 0im)/√2, ξy = (-1im)/√2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))

# p = 1
MCPLGTYPE10 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = 0, p = 1,  ξx = (1. + 0im)/√2, ξy = (-1im)/√2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))
MCPLGTYPE1_1 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = -1, p = 1,  ξx = (1. + 0im)/√2, ξy = (-1im)/√2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))
MCPLGTYPE11 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = 1, p = 1,  ξx = (1. + 0im)/√2, ξy = (-1im)/√2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))


################################################################################
#Linearly Polarized

# p = 0
LPLGTYPE00 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = 0, p = 0, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))
LPLGTYPE0_1 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = -1, p = 0, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))
LPLGTYPE01 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = 1, p = 0, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))

# p = 1
LPLGTYPE10 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = 0, p = 1, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))
LPLGTYPE1_1 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/4,
    w₀ = w₀, m = -1, p = 1, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))
LPLGTYPE11 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/4,
    w₀ = w₀, m = 1, p = 1, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))

# p = 2
LPLGTYPE20 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = 0, p = 2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))
LPLGTYPE21 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = 1, p = 2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))
LPLGTYPE22 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = 2, p = 2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))
LPLGTYPE2_1 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = -1, p = 2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))
LPLGTYPE2_2 = LaguerreGaussLaser(c = c, m_q = 1., q = -1., λ = λ, a₀ = a₀, # ϕ₀ = -π/2,
    w₀ = w₀, m = -2, p = 2, profile = ConstantProfile()) # GaussProfile(c = c, τ = w/ω, z₀ = 0.))

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

const domainXY = range(-maxR, stop = maxR, length = 800)/w₀
const domainR  = range(0., stop = maxR, length = 800)/w₀
const time = range(- n * τ, stop = n * τ, length = 1200)/T₀

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

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
        plot!(underlay, X, Y; kwargs...) 
end

function polarization(p)
    if imag(p.type.ξy)*√2 == 1.
        println("Right Circular Polarization, p = $(p.type.p), m = $(p.type.m)")
        return "right-handedCP"
    end
    if imag(p.type.ξy)*√2 == -1.
        println("Left Circular Polarization, p = $(p.type.p), m = $(p.type.m)")
        return "left-handedMCP"
    end
    if imag(p.type.ξy)*√2 == 0.
        println("Linear Polarization, p = $(p.type.p), m = $(p.type.m)")
        return "linearLP"
    end
end

for (sim_name, p) in pairs(field_param)
    folder_setup("$sim_name")
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

    jz₀(x,y,t = 0) = ([x,y,0.]×(fieldE(x,y,0,t)×fieldB(x,y,0,t)))[3]
    jzρ(ρ) = quadgk(ϕ -> jz₀(ρ*cos(ϕ), ρ*sin(ϕ))/(2*π), 0, 2*π, rtol = 1e-8, atol = 1.e-5)[1]

    # end of definitions
    
    # plots start here

    plot_data = [plot([], [], label = false) for i in 1:8]

    # 1. Heatmaps

    common_settings = (fontfamily = "computer modern", dpi = 360, label = false)
    plot_settings   = (color_palette = :Set1_4, seriescolor = 4, linewidth = 2.5)

    #  (x,y) -> (jz₀(w₀*x, w₀*y) + 0.00001)/(wenergy₀(w₀*x, w₀*y) + 0.00001)
    jz_period(x, y) = quadgk(t -> jz₀(x, y, t), 0, T₀, rtol = 1e-8, atol = 1e-8)[1]

    # plot_data[8] = heatmap(domainXY, domainXY, 
    #             (x,y) -> jz_period(w₀*x, w₀*y)/wenergy_period(w₀*x, w₀*y); #(any([(abs(hypot(x,y)-root) < 0.05) for root in p.roots]) ? (p.type.m + imag(p.type.ξy)*√2)/ω : jz₀(w₀*x, w₀*y, T₀/4)/wenergy(w₀*x, w₀*y, T₀/4));
    #             xlabel = L"x/w_0", ylabel = L"y/w_0", 
    #             color_palette = :lajolla,
    #             aspect_ratio = 1, common_settings...)
    # add_info!(plot_data[8], p, true) 
    # savefig(plot_data[8], "j_w$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")
    
    plot_data[1] = heatmap(domainXY, domainXY, (x,y) -> wenergy₀(w₀*x, w₀*y);
                xlabel = L"x/w_0", ylabel = L"y/w_0", 
                color_palette = :lajolla,
                aspect_ratio = 1, common_settings...)
    add_info!(plot_data[1], p, true) 
    savefig(plot_data[1], "xy00Energy$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")
    println("Energy density for time = 0, z = 0, in the xy-plane")

    plot_data[2] = heatmap(domainXY, domainXY, (x,y) -> xyEx(w₀*x, w₀*y);
                xlabel = L"x/w_0", ylabel = L"y/w_0",
                color_palette = :vik, 
                aspect_ratio = 1, common_settings...)
    add_info!(plot_data[2], p, true)
    savefig(plot_data[2], "xy00Ex$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")
    println("Electric field on the x-axis for time = 0, z = 0, in the xy-plane")

    plot_data[3] = heatmap(domainXY, domainXY, (x,y) -> xyEz(w₀*x, w₀*y);
                xlabel = L"x/w_0", ylabel = L"y/w_0", 
                color_palette = :vik, 
                aspect_ratio = 1, common_settings...)
    add_info!(plot_data[3], p, true)
    savefig(plot_data[3], "xy00Ez$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")
    println("Electric field on the z-axis for time = 0, z = 0, in the xy-plane")

    plot_data[4] = heatmap(domainXY, domainXY, (x,y) -> jz₀(w₀*x, w₀*y);
                xlabel = L"x/w_0", ylabel = L"y/w_0",
                color_palette = :vik, 
                aspect_ratio = 1, common_settings...)
    add_info!(plot_data[4], p, true)
    savefig(plot_data[4], "xy00jz$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")
    println("Field angular momentum density on the z-axis for time = 0, z = 0, in the xy-plane")

    # 2. Azimuthally averaged radial plots

    jzρ_ = [jzρ(r*w₀) for r in domainR]
    plot_data[5] = plot(domainR, jzρ_, label = false)
    plot_data[5] = add_rect!(plot_data[5], domainR, jzρ_, p.roots;
                xlabel = L"\rho/w_0", ylabel = L"\overline{j_z^{xy}}(\rho, t = 0)\quad\mathrm{[a.u.]}", 
                common_settings..., plot_settings...)
    add_info!(plot_data[5], p)
    savefig(plot_data[5], "rjz$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")
    println("Azimuthally averaged angular momentum on the z-axis for time = 0, z = 0, in the xy")
    
    wρ_ = [wρ(r*w₀) for r in domainR]
    plot_data[6] = plot(domainR, wρ_, label = false)
    plot_data[6] = add_rect!(plot_data[6], domainR, wρ_, p.roots; 
        xlabel = L"\rho/w_0", ylabel = L"\mathrm{Intensity}(\rho)\quad\mathrm{[a.u.]}",
        common_settings..., plot_settings...)
    add_info!(plot_data[6], p)
    savefig(plot_data[6], "rw$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")
    println("Azimuthally averaged energy density for time = 0, z = 0, in the xy")

    # 3. Time axis plots.

    tE_ = [fieldE(w₀/√2/2, w₀/√2/2, 0, ti*T₀)[1] for ti in time]
    plot_data[7] = plot(time, tE_;
         xlabel = L"t/T_0", ylabel = L"E_x({w_0}/{\sqrt{8}}, {w_0}/{\sqrt{8}}, 0, t)\quad\mathrm{[a.u.]}",
         common_settings..., plot_settings...)
    add_info!(plot_data[7], p)
    savefig(plot_data[7],"tE$(imag(p.type.ξy)*√2)$(p.type.p)$(p.type.m).png")
    println("Electric field on the x-axis in time for x = w₀/√2/2, y = w₀/√2/2, z = 0")

    serialize("plot_data_$(pol)_$(p.type.p)_$(p.type.m)", plot_data)
    cd("..")
end
