const g = @SMatrix [1  0  0  0;
                    0 -1  0  0;
                    0  0 -1  0;
                    0  0  0 -1] # Minkowski space metric

function dynEq(v,type,τ)
    x = v[1:4]
    u = v[5:8]

    q = type.constants.q
    mₑ = type.constants.mₑ

    dx = u
    du = q/mₑ * Fμν(x, type) * g * u

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
    [Mμν(xτ) for xτ in X.u]
end

function UniformAnnulus(r₁,r₂)
    θ = rand(Uniform(0,2*π))
    u = rand(Uniform(r₁,r₂))
    v = rand(Uniform(0,r₁+r₂))
    ρ = v<u ? u : r₁+r₂ - u
    [SVector{3}(ρ*cos(θ), ρ*sin(θ),0), ρ]
end

R = [UniformAnnulus(0,maxR) for i in 1:N] 
RVec = VectorOfArray(R)
idx = sortperm(R, lt=(x,y)->x[2]<y[2])
R₀ = RVec[1,:][idx]
ρ₀ = RVec[2,:][idx]

function xu_cat(X,U,T)
    x0 = c*T
    x = vcat([x0],X)
    u0 = √(c*c + U⋅U)
    u = vcat([u0],U)
    SVector{8}(vcat(x,u))
end
        
function problem_set(R₀, p, U₀ = [0,0,0])
    V₀ = [xu_cat(r₀,U₀,τi) for r₀ in R₀]
    prob = ODEProblem(dynEq, V₀[1], (τi, τf), p)
    
    function prob_func(prob,i,repeat)
        ODEProblem(dynEq, V₀[i], (τi, τf), p) #return
    end
    
    function output_func(sol, i)
        (Mμντ(sol), false) # put sol as argument, if you want the actual trajectories
    end
    
    EnsembleProblem(prob; prob_func, output_func)
end

# new
function Moment_Avg(M, J; moment_func = identity, f = 1, l = N)
    N_samples = 1 + l - f
    func(i,t) = moment_func(getproperty.((M[i][t], ), J)...)
    [ThreadsX.sum(func(i,t) for i in f:l)/N_samples for t in 1:time_samples+1]
end

# function Radial_Avg(M, args_funcs, bins = 100) # J = (:Lz, (:x¹, :x²)) 
# data_points = zeros(bins, 1 + sum(length.(args_funcs)) - length(args_funcs)) 

# particle_idx = 1
# current_M = M[particle_idx][end] # current_particle data at the final time
# current_ρ = ρ₀[particle_idx] # ⟺ hypot(m[begin].x¹, m[begin].x²), but faster since in memory already

#     for i in 1:bins
#         bin_averages = zeros(1 + 2length(args_funcs))
#         tally = 0
#         while current_ρ < i * maxR/bins
#             bin_averages[1]     += current_ρ
#             bin_averages[2:end] += [func(getproperty.((current_M,), args[begin])...) 
#                                                         for args in args_funcs 
#                                                             for func in args[2:end]]
#             tally += 1
#             try 
#                 Midx += 1
#                 current_M = M[particle_idx][end]
#                 current_ρ = ρ₀[particle_idx] 
#             catch err
#                 # isa(err, BoundsError)
#                 break
#             end 
#         end
#         data_points[i,:] = bin_averages/tally
#     end  
#     return data_points
# end

# args_funcs = [((:Lz,), identity, x -> x^2), ((:x¹, :x²), hypot, (x,y) -> x^2 + y^2)]

function Radial_Avg(M, bins = 100)
    LzStats = zeros(bins, 3)
    ρStats  = zeros(bins, 3) 
    Midx = 1
    m = M[Midx][end]
    ρ = ρ₀[Midx]
    for i in 1:bins
        ρ_avg, Lz_, Lz²_, ρ_, ρ²_ = 0., 0., 0., 0., 0.  
        tally = 0
        while ρ < i * maxR/bins
            ρ_avg += ρ 
            Lz_ += m.Lz  
            Lz²_ += m.Lz^2
            ρ_ += hypot(m.x¹, m.x²)
            ρ²_ += m.x¹^2 + m.x²^2  
            tally += 1
            try Midx < N
                Midx += 1
                m = M[Midx][end]
                ρ = ρ₀[Midx]
            catch err
                break 
            end 
        end
        rt = ρ_avg/tally
        LzStats[i,:] .= rt, Lz_/tally, Lz²_/tally
        ρStats[i,:] .= rt, ρ_/tally,  ρ²_/tally
    end  
    return LzStats, ρStats
end

common_settings = (fontfamily = "computer modern", 
                   dpi = 360, 
                   label = false, 
                   fillalpha=.3, 
                   color_palette = :Set1_4,
                   linewidth = 2)

for type in lasers
    pol = polarization(type)
    
    folder_setup("$(pol)_p$(type.p)_m$(type.m)") # change folder for each simulation
    folder_setup("particles") # separate particles from fields
    
    file_ext = "$(pol)_p$(type.p)_m$(type.m).$fmt"
    
    write(io, "$(pol), p = $(type.p), m = $(type.m)\n")

    plot_data = [plot([],[], label = false) for i in 1:8]

    eprob = problem_set(R₀, type)
    M = solve(eprob, Vern9(), EnsembleThreads(), abstol = 1e-15, reltol = 1e-15,
            saveat = (τf-τi)/time_samples, trajectories = N)

    roots = rootsLUT[(p = type.p, m = abs(type.m))]
    idxs = zeros(Int, length(roots))                   # initialize index list

    root_idx = 2                                       # first root is always zero, start from root 2
    for (i,v) in enumerate(ρ₀)                         # ρ₀ is ordered!  
        try
            if v > w₀*roots[root_idx]
                idxs[root_idx] = i-1                   # take the last one which respects the condition
                root_idx += 1
            end
        catch err
            # isa(err, BoundsError)
            break
        end
    end

    for j in 2:(length(roots))       
        z,  Lz,  L² = Moment_Avg.((M,), (:x³, :Lz, :L²); f = 1+idxs[j-1], l = idxs[j])
        z², Lz², L⁴ = Moment_Avg.((M,), (:x³, :Lz, :L²); moment_func = x -> x^2, f = 1+idxs[j-1], l = idxs[j])
        ρ  = Moment_Avg(M, (:x¹, :x²); moment_func = hypot, f = 1+idxs[j-1], l = idxs[j])
        # ρ² = Moment_Avg(M, (:x¹, :x²); moment_func = (x, y) -> x^2 + y^2, f = 1+idxs[j-1], l = idxs[j])
        
        σz  = σA(z, z²)/w₀
        # σρ  = σA(ρ, ρ²)/w₀ # not worth computing because it's too wide and breaks plots
        σLz = σA(Lz, Lz²)/w₀/π₀
        σL² = σA(L², L⁴)/w₀/π₀/w₀/π₀

        write(io, 
            "from $(roots[j-1]/w₀) to $(roots[j]/w₀)\n"*
            "z ± σz $(z[end]/w₀) ± $(σz[end])\n"*
            "Lz ± σLz $(Lz[end]/w₀/π₀) ± $(σLz[end])\n"*
            "L² ± σL² $(L²[end]/w₀/π₀/w₀/π₀) ± $(σL²[end])\n"
             )

        plot_data[1] = plot!(plot_data[1], Δτ, z/λ; 
                                common_settings...,
                                ribbon = σz, 
                                seriescolor = j - 1, # these cannot be moved outside due to variable scope
                                xlabel = L"\tau/T_0", 
                                ylabel = L"\overline{z}/\lambda")

        # the average radius for particles homogeneously distributed in the annulus between root[i] and root[i+1]                 
        mean_ρ = 2/3*(roots[j]^3 - roots[j-1]^3)/(roots[j]^2 - roots[j-1]^2)

        plot_data[2] = plot!(plot_data[2], Δτ, ρ/w₀ .- mean_ρ; 
                                common_settings..., 
                                # ribbon = σρ, too big relative to ρ changes, makes plot unreadable               
                                seriescolor = j - 1,
                                xlabel = L"\tau/T_0",
                                ylabel = L"\Delta\overline{\rho}/w_0")

        plot_data[3] = plot!(plot_data[3], Δτ, Lz/w₀/π₀;
                                common_settings..., 
                                ribbon = σLz,
                                seriescolor = j - 1,
                                xlabel = L"\tau/T_0",
                                ylabel = L"\overline{L_z}/(w_0 \pi_0)")

        plot_data[4] = plot!(plot_data[4], Δτ, L²/w₀/π₀/w₀/π₀;
                                common_settings...,
                                ribbon = (L²/w₀/π₀/w₀/π₀, σL²),   
                                seriescolor = j - 1,
                                xlabel = L"\tau/T_0",
                                ylabel = L"\overline{L^2}/(w_0^2 \pi_0^2)")

    end
    write(io,"\n")

    add_info!(plot_data[1], type)
    savefig(plot_data[1], "z_t_$file_ext")
    add_info!(plot_data[2], type)
    savefig(plot_data[2], "rho_t_$file_ext")
    add_info!(plot_data[3], type)
    savefig(plot_data[3], "Lz_t_$file_ext")
    add_info!(plot_data[4], type)
    savefig(plot_data[4], "L2_t_$file_ext")

    X = [M[i][begin].x¹ for i in 1:N]/w₀ # pulling out the initial x coordinates
    Y = [M[i][begin].x² for i in 1:N]/w₀ # pulling out the initial y coordinates    
    Lz = [M[i][end].Lz for i in 1:N] # pulling out the final Lz for colouring   
    Lzmax = max(abs.(extrema(Lz))...) # finding maximum
    # colours = Lz/Lzmax # doing a normalization

    write(io, "Maximum Lz transfered = $Lzmax\n\n")
    
    plot_data[7] = scatter(X,Y;
        marker_z = Lz,
        markershape = circ,
        markerstrokealpha = 0,
        markersize = 1.2,
        seriescolor = cgrad(:PuOr_9, rev = true),
        clims = (-Lzmax, Lzmax),
        fontfamily = "computer modern", 
        dpi = 360, 
        label = false,  
        xlabel = L"x/w_0", ylabel = L"y/w_0", 
        aspect_ratio = 1,
        xticks = -4:1:4,
        right_margin = 0.9*Plots.cm,)
    add_info!(plot_data[7], type, true)
    savefig(plot_data[7], "XYLz_$file_ext")

    @time begin
    particle_range = 1:50:N
    Xt  = [[M[i][t].x¹/w₀ for t in 1:time_samples] for i in particle_range]
    Yt  = [[M[i][t].x²/w₀ for t in 1:time_samples] for i in particle_range]
    Zt  = [[M[i][t].x³/λ  for t in 1:time_samples] for i in particle_range]
    Lzt = [[M[i][t].Lz    for t in 1:time_samples] for i in particle_range]
    plot_data[8] = plot(Xt, Yt, Zt;
        line_z = VectorOfArray(Lzt),
        right_margin = 0.9*Plots.cm,
        seriescolor = cgrad(:PuOr_9, rev = true),
        xlabel = L"x/w_0", 
        ylabel = L"y/w_0",
        zlabel = L"z/\lambda",
        common_settings...)
    # add_info!(plot_data[8], type) broken for 3D plots
    savefig(plot_data[8], "XYZ_$file_ext")
    end

    LzS, ρS = Radial_Avg(M)
    
    plot_data[5] = plot(LzS[:,1]/w₀, LzS[:,2]/π₀/w₀, 
                        label = false,
                        ribbon = σA(LzS[:,2], LzS[:,3])/π₀/w₀)
    plot_data[5] = add_rect!(plot_data[5], LzS[:,1]/w₀, LzS[:,2]/π₀/w₀, roots;
                        common_settings...,
                        ribbon = σA(LzS[:,2], LzS[:,3])/π₀/w₀,
                        seriescolor = 4,
                        xlabel = L"\rho_0/w_0", 
                        ylabel = L"\overline{L_z}(\rho_0)/(w_0 \pi_0)",
                        )
    add_info!(plot_data[5], type)
    savefig(plot_data[5], "Lz_rho_$file_ext")
    
    plot_data[6] = plot(ρS[:,1]/w₀, ρS[:,2]/w₀ .- ρS[:,1]/w₀;
                        label = false,
                        ribbon = σA(ρS[:,2], ρS[:,2])/w₀)
    plot_data[6] = add_rect!(plot_data[6], 
                        ρS[:,1]/w₀, ρS[:,2]/w₀ .- ρS[:,1]/w₀, roots;
                        common_settings...,
                        ribbon = σA(ρS[:,2], ρS[:,3])/w₀,
                        seriescolor = 4,
                        xlabel = L"\rho_0/w_0", 
                        ylabel = L"\Delta\overline{\rho}_{\mathrm{final}}(\rho_0)/w_0")
    add_info!(plot_data[6], type)
    savefig(plot_data[6], "rho_rho_$file_ext")

    serialize("plots.jls", plot_data)
    serialize("radial_ρ_data.jls", LzS)
    serialize("radial_Lz_data.jls", LzS)
    # serialize("plot_data.jls", M) save at your own peril, too large to be read efficiently
    cd("../..") # return to output_* base folder 
end
