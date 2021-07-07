const domainXY = range(-maxR, stop = maxR, length = 333)/w₀
const domainR  = range(0., stop = maxR, length = 800)/w₀
const time = range(- n * τ, stop = n * τ, length = 1200)/T₀

# useful function definitions

function w_inst(x, y, z, t, type) 
    @unpack ε₀, c = fundamental_constants(type)
    return (ε₀/2)*(E([x,y,z], t, type)⋅E([x,y,z], t, type) + 
               c^2*B([x,y,z], t, type)⋅B([x,y,z], t, type))
end

# dot takes the conjugate of the first vector automatically!
function w_cycle(x, y, z, type) 
    @unpack ε₀, c = fundamental_constants(type)
    return (ε₀/4)*real(E([x, y, z], type)⋅E([x, y, z], type) + 
                 (c^2)*B([x, y, z], type)⋅B([x, y, z], type))
end
                              
function j_inst(x, y, z, t, type) 
    ε₀ = fundamental_constants(type, :ε₀)
    return ε₀*([x,y,z]×(E([x,y,z], t, type)×B([x,y,z], t, type)))
end

function jz_cycle(x, y, z, type) 
    ε₀ = fundamental_constants(type, :ε₀)

    return (ε₀/4)*real((x*(
                            (E([x, y, z], type)×conj(B([x, y, z], type)))[2] + 
                            (conj(E([x, y, z], type))×B([x, y, z], type))[2]
                            ) 
                            - 
                        y*(
                            (E([x, y, z], type)×conj(B([x, y, z], type)))[1] + 
                            (conj(E([x, y, z], type))×B([x, y, z], type))[1]
                            ))
                            )
end

function Mz(x, y, z, t, type) # normed to the q*E₀*w₀
    q = type.constants.q
    E₀ = type.derived.E₀
    w₀ = type.w₀
    sign(q)*(x*E([x, y, z], t, type)[2] - y*E([x, y, z], t, type)[1])/(E₀*w₀)
end

E_ρAvg(ρ, z, t, type) = quadgk(ϕ -> E([ρ*cos(ϕ), ρ*sin(ϕ), z], t, type)/(2*π), 0, 2*π, rtol = 1e-8, atol = 1e-10)[1]

wi_ρAvg(ρ, z, t, type) = quadgk(ϕ -> w_inst(ρ*cos(ϕ), ρ*sin(ϕ), z, t, type)/(2*π), 0, 2*π, rtol = 1e-8)[1]
wc_ρAvg(ρ, z, type) = quadgk(ϕ -> w_cycle(ρ*cos(ϕ), ρ*sin(ϕ), z, type)/(2*π), 0, 2*π, rtol = 1e-8)[1]

ji_ρAvg(ρ, z, t, type) = quadgk(ϕ -> j_inst(ρ*cos(ϕ), ρ*sin(ϕ), z, t, type)/(2*π), 0, 2*π, rtol = 1e-8, atol = 1e-10)[1]
jc_ρAvg(ρ, z, type) = quadgk(ϕ -> jz_cycle(ρ*cos(ϕ), ρ*sin(ϕ), z, type)/(2*π), 0, 2*π, rtol = 1e-8, atol = 1e-10)[1]

# end of definitions

common_settings = (fontfamily = "computer modern", dpi = 360, label = false)
plot_settings   = (color_palette = :Set1_4, seriescolor = 4, linewidth = 2.5)

for type in lasers
    pol = polarization(type)

    folder_setup("$(pol)_p$(type.p)_m$(type.m)") # change folder for each simulation
    folder_setup("fields") # separate fields from particles

    # w₀ = type.w₀
    p = type.p 
    m = type.m
    ξy = type.polarization.ξy
    # T₀ = type.derived.T₀
    roots = rootsLUT[(p = p, m = abs(m))]

    file_ext = "$(pol)_p$(p)_m$(m).$(fmt)"
    
    # 1. Heatmaps
    
    w_00 = heatmap(domainXY, domainXY, (x,y) -> w_inst(w₀*x, w₀*y, 0, 0, type);
                xlabel = L"x/w_0", ylabel = L"y/w_0", 
                seriescolor = :linear_kryw_5_100_c67_n256,
                clims = x -> (0, maximum(x)),
                aspect_ratio = 1, common_settings...)
    add_info!(w_00, type, true) 
    savefig(w_00, "xy00Energy_$file_ext")
    println("Energy density for time = 0, z = 0, in the xy-plane")

    w_0T₀ = heatmap(domainXY, domainXY, (x,y) -> w_cycle(w₀*x, w₀*y, 0, type);
                xlabel = L"x/w_0", ylabel = L"y/w_0", 
                seriescolor = :linear_kryw_5_100_c67_n256,
                clims = x -> (0, maximum(x)),
                aspect_ratio = 1, common_settings...)
    add_info!(w_0T₀, type, true) 
    savefig(w_0T₀, "xy0T₀Energy_$file_ext")
    println("Energy density cycle averaged, z = 0, in the xy-plane")

    Ex_00 = heatmap(domainXY, domainXY, (x,y) -> E([w₀*x, w₀*y, 0], 0, type)[1];
                xlabel = L"x/w_0", ylabel = L"y/w_0",
                seriescolor = :vik,
                clims = x -> maximum(abs.(extrema(x))).*(-1,1), 
                aspect_ratio = 1, common_settings...)
    add_info!(Ex_00, type, true)
    savefig(Ex_00, "xy00Ex_$file_ext")
    println("Electric field on the x-axis for time = 0, z = 0, in the xy-plane")

    Ez_00 = heatmap(domainXY, domainXY, (x,y) -> E([w₀*x, w₀*y, 0], 0, type)[3];
                xlabel = L"x/w_0", ylabel = L"y/w_0", 
                seriescolor = :vik,
                clims = x -> maximum(abs.(extrema(x))).*(-1,1), 
                aspect_ratio = 1, common_settings...)
    add_info!(Ez_00, type, true)
    savefig(Ez_00, "xy00Ez_$file_ext")
    println("Electric field on the z-axis for time = 0, z = 0, in the xy-plane")

    jz_00 = heatmap(domainXY, domainXY, (x,y) -> j_inst(w₀*x, w₀*y, 0, 0, type)[1];
                xlabel = L"x/w_0", ylabel = L"y/w_0",
                seriescolor = :vik,
                clims = x -> maximum(abs.(extrema(x))).*(-1,1), 
                aspect_ratio = 1, common_settings...)
    add_info!(jz_00, type, true)
    savefig(jz_00, "xy00jz_$file_ext")
    println("z-axis field angular momentum density for time = 0, z = 0, in the xy-plane")

    jz_0T₀ = heatmap(domainXY, domainXY, (x,y) -> jz_cycle(w₀*x, w₀*y, 0, type); 
                xlabel = L"x/w_0", ylabel = L"y/w_0", 
                seriescolor = :vik,
                clims = x -> maximum(abs.(extrema(x))).*(-1,1),
                aspect_ratio = 1, common_settings...)
    add_info!(jz_0T₀, type, true) 
    savefig(jz_0T₀, "xy0T₀jz_$file_ext")
    println("z-axis field angular momentum density cycle averaged, z = 0, in the xy-plane")

    jz_w = heatmap(domainXY, domainXY, (x,y) -> ω*jz_cycle(w₀*x, w₀*y, 0, type)/w_cycle(w₀*x, w₀*y, 0, type); 
                xlabel = L"x/w_0", ylabel = L"y/w_0", 
                seriescolor = :lajolla,
                aspect_ratio = 1, common_settings...)
    add_info!(jz_w, type, true) 
    savefig(jz_w, "xy0T₀jz_w_$file_ext")
    println("z-axis field angular momentum density divided by energy density, cycle averaged, z = 0, in the xy-plane")

    Lz′ = heatmap(domainXY, domainXY, (x,y) -> Mz(w₀*x, w₀*y, 0, 0, type); 
                xlabel = L"x/w_0", ylabel = L"y/w_0", 
                seriescolor = :vik,
                clims = x -> maximum(abs.(extrema(x))).*(-1,1),
                aspect_ratio = 1, common_settings...)
    add_info!(Lz′, type, true) 
    savefig(Lz′, "xy00Lz_$file_ext")
    println("Instantaneous reduced torque for time t = 0, z = 0, in the xy-plane")

    @time begin
    Lz′_avg = heatmap(domainXY, domainXY, (x,y) -> quadgk(t -> Mz(w₀*x, w₀*y, 0, t, type), -Inf, Inf, atol = 1e-2)[1]; 
    xlabel = L"x/w_0", ylabel = L"y/w_0", 
    seriescolor = :vik,
    clims = x -> maximum(abs.(extrema(x))).*(-1,1),
    aspect_ratio = 1, common_settings...)
    add_info!(Lz′_avg, type, true) 
    savefig(Lz′_avg, "xy0∞Lz_$file_ext")
    println("Pulse averaged reduced torque for, z = 0, in the xy-plane")
    end
    # 2. Azimuthally averaged radial plots

    jz_ρAvg = plot(domainR, r -> jc_ρAvg(w₀*r, 0, type); 
                label = false)
    jz_ρAvg = add_rect!(jz_ρAvg, domainR, r -> jc_ρAvg(w₀*r, 0, type), roots;
                xlabel = L"\rho/w_0", ylabel = L"\overline{j_z^{xy}}(\rho, t = 0)\quad\mathrm{[a.u.]}", 
                common_settings..., plot_settings...)
    add_info!(jz_ρAvg, type)
    savefig(jz_ρAvg, "jz_ρAvg_$file_ext")
    println("Azimuthally averaged angular momentum on the z-axis for time = 0, z = 0, in the xy")
    
    w_rAvg = plot(domainR, r -> wc_ρAvg(w₀*r, 0, type); 
                label = false)
    w_rAvg = add_rect!(w_rAvg, domainR, r -> wc_ρAvg(w₀*r, 0, type), roots; 
        xlabel = L"\rho/w_0", ylabel = L"\mathrm{Intensity}(\rho)\quad\mathrm{[a.u.]}",
        common_settings..., plot_settings...)
    add_info!(w_rAvg, type)
    savefig(w_rAvg, "w_ρAvg_$file_ext")
    println("Azimuthally averaged energy density for time = 0, z = 0, in the xy")

    # 3. Time axis plots.

    Ex_t = plot(time, t -> E([w₀/√2/2, w₀/√2/2, 0], t*T₀, type)[1];
         xlabel = L"t/T_0", ylabel = L"E_x({w_0}/{\sqrt{8}}, {w_0}/{\sqrt{8}}, 0, t)\quad\mathrm{[a.u.]}",
         common_settings..., plot_settings...)
    add_info!(Ex_t, type)
    savefig(Ex_t,"Ex_t_$file_ext")
    println("Electric field on the x-axis in time for x = w₀/√2/2, y = w₀/√2/2, z = 0")

    # 3.1 Animations
   
    animLz = Animation()
    for (i,t) in enumerate(range(-n*T₀, n*T₀, length = 299)) # +1 = 300 / 30fps = 10s
        Lz′_anim = heatmap(domainXY, domainXY, (x,y) -> Mz(w₀*x, w₀*y, 0,  t, type); 
                    xlabel = L"x/w_0", ylabel = L"y/w_0", 
                    seriescolor = :vik,
                    clims = x -> (-0.5,0.5), # maximum(abs.(extrema(x))).*(-1,1),
                    cbar = :none,
                    aspect_ratio = 1, 
                    fontfamily = "computer modern", 
                    dpi = 120, 
                    label = false)
        add_info!(Lz′_anim, type, true) 
        frame(animLz)
        pritnln("Animatin ... Frame $i/300.")
    end
    gif(animLz, "Lz′.gif", fps = 30)
    println("Instantaneous reduced torque for time t = 0, z = 0, in the xy-plane")

    cd("../..") # return to output_* base folder 
    break
end
