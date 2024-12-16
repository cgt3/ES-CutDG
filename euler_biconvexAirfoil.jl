
using LinearAlgebra
using OrdinaryDiffEq
using RecursiveArrayTools
using StaticArrays
using StructArrays

using PathIntersections
using StartUpDG
using Trixi

include("alive.jl")
include("es_rhs.jl")

# Simulation parameters
domain = (; x_lb=-0.5, x_ub=5.5, y_lb=-1.5, y_ub=1.5)
# objects = (PresetGeometries.Circle(R=0.331, x0=0.0),)
objects = (PresetGeometries.BiconvexAirfoil(scale=0.5, x0=0.03),)

# Get the flux functions from Trixi
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)
# equations = ShallowWaterEquations2D(gravity_constant=1.0)

# Volume fluxes
fs_x(UL, UR) = flux_ranocha(UL, UR, 1, equations)
fs_y(UL, UR) = flux_ranocha(UL, UR, 2, equations)


# Boundary flux
# For entropy stability:
fs_boundary(UL, UR, n) = flux_lax_friedrichs(UL, UR, n, equations)

fs = (; fs_x, fs_y, fs_boundary)
const num_fields = 4
const num_dim = 2


rho_inf = 20.0
p_inf   = 20.0
c0 = sqrt(gamma*p_inf/rho_inf)

v1_max =  1.5*sqrt(gamma*rho_inf/p_inf)


function uIC(x, y)
    return prim2cons(SVector{num_fields}([rho_inf, v1_max, 0.0, p_inf]), equations )
end

# Set the boundary conditions and forcing
function BC(x,y,t, nx, ny, uf)
    # Reflective BCs on top and btm
    if abs(y - domain.y_lb) < 1e-10 || abs(y - domain.y_ub) < 1e-10
        v_n = uf[2]*nx + uf[3]*ny
        return SVector{4, Float64}(uf[1], uf[2] - 2*v_n*nx, uf[3] - 2*v_n*ny, uf[4])
        # return uf
    # Velocity-inlet on the left
    elseif abs(x - domain.x_lb) < 1e-10
        # v1 = v1_prescribed(x,y,t)
        return prim2cons( SVector{num_fields}([rho_inf, v1_max, 0.0, p_inf]), equations )

    # Pressure outlet on the right
    # The pressure used must decrease 
    elseif abs(x - domain.x_ub) < 1e-10
        return uf
    else  
        # Reflective on the airfoil
        v_n = uf[2]*nx + uf[3]*ny
        return SVector{4, Float64}(uf[1], uf[2] - 2*v_n*nx, uf[3] - 2*v_n*ny, uf[4])
    end
end

function forcing(u, x,y,t)
    return SVector{4, Float64}(0.0*x,  0.0*x, 0.0*x, 0.0*x)
end

N_deg = 4
# Fine mesh: 20*
cells_per_dimension_x = Integer(8*ceil(domain.x_ub - domain.x_lb))
cells_per_dimension_y = Integer(8*ceil(domain.y_ub - domain.y_lb))
coordinates_min = (domain.x_lb, domain.y_lb)
coordinates_max = (domain.x_ub, domain.y_ub)

rd = RefElemData(Quad(), Polynomial{Gauss}(), N=N_deg)
md = MeshData(rd,
            objects,
            cells_per_dimension_x, 
            cells_per_dimension_y;
            coordinates_min=coordinates_min, 
            coordinates_max=coordinates_max,
            precompute_operators=true)

# Apply the initial condition
u = uIC.(md.x, md.y)
u_float = NamedArrayPartition((; cartesian=unwrap_array(u.cartesian), 
                                cut=unwrap_array(u.cut) ))

# SRD if needed/desired
srd = StateRedistribution(rd, md, eltype(u.cut))
# srd = 0

# Generate helper memory and the hybridized SBP operators for use in rhs!
memory = allocate_rhs_memory(md)
cartesian_operators, cut_operators = generate_operators(rd, md)

params = (; cartesian_operators, cut_operators, BC, forcing, fs, md, rd, equations, memory, use_srd=true, use_entropy_vars=false, srd)

t_start = 0
t_end = 5
t_span = (t_start, t_end)
t_save = LinRange(t_span..., Integer(ceil(t_end - t_start))*100 + 1)

# Simulate the PDE
experiment="impulsiveStart_coarseMesh_troubleshoot6"
prob = ODEProblem(rhs!, u_float, t_span, params)
sol = solve(prob, Tsit5(), adaptive=true, abstol=1e-8, reltol=1e-6, dt=1e-8, saveat=t_save, callback=AliveCallback(alive_interval=20));

# using FileIO, JLD2
# FileIO.save(@sprintf("savedSoln/euler_biconvex_%s.jld2", experiment), 
#     "sol", sol,
#     "rd", rd,
#     "md", md,
#     "v1_max", v1_max,
#     "p_inf", p_inf,
#     "rho_inf", rho_inf
# )

include("helperCodes/Plotting_CutMeshes.jl")

x_plot = LinRange(domain.x_lb, domain.x_ub, Integer( ceil(200*(domain.x_ub-domain.x_lb)) ))
y_plot = LinRange(domain.y_lb, domain.y_ub, Integer( ceil(200*(domain.y_ub-domain.y_lb)) ))
V_plot = global_interpolation_op(x_plot, y_plot, u_float, domain, md, rd)


function density(u, x, y, t)
    return u[1,:]
end

function x_velocity(u, x, y, t)
    return u[2,:] ./ u[1,:]
end

function pressure(u, x, y, t)
    u_prim = similar(u)
    for i in axes(u_prim,2)
        u_prim[:,i] = cons2prim(u[:,i], equations)
    end
    return u_prim[4,:]
end

function titleString(t)
    return @sprintf("t=%.2lf", t)
end

plotting_increment = 5;
fps = 20

u_plot = sol.u

## Make GIF with a grid
domain_plot = (; x_lb=-0.5, x_ub=0.5, y_lb=-0.5, y_ub=0.5)
makeGIF_grid(u_plot, t_save, density, length(sol.u), md, rd, x_plot, y_plot, V_plot, domain, 
    t_step_incr=plotting_increment,
    plot_lims = (0.7*rho_inf, 1.3*rho_inf),
    filename=@sprintf("eulerBiconvex_gridPlot_density_%s.mp4", experiment),
    fps = fps,
    sol_color=cgrad(:PuOr, rev=true), #:Purples,
    plot_embedded_objects=true,
    line_color=:black,
    titleString=titleString,
)

makeGIF_grid(u_plot, t_save, pressure, length(sol.u), md, rd, x_plot, y_plot, V_plot, domain, 
    t_step_incr=plotting_increment,
    plot_lims = (0.6*p_inf, 1.4*p_inf),
    filename=@sprintf("eulerBiconvex_gridPlot_pressure_%s.mp4", experiment),
    fps = fps,
    sol_color=cgrad(:RdGy, rev=true),
    plot_embedded_objects=true,
    line_color=:black,
    titleString=titleString,
)

makeGIF_grid(u_plot, t_save, x_velocity, length(sol.u), md, rd, x_plot, y_plot, V_plot, domain, 
    t_step_incr=plotting_increment,
    plot_lims = (0.8*v1_max, 1.2*v1_max),
    filename=@sprintf("eulerBiconvex_gridPlot_xVelocity_%s.mp4", experiment),
    fps = fps,
    sol_color=:Spectral,
    plot_embedded_objects=true,
    line_color=:black,
    titleString=titleString,
)
