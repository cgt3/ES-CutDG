
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

domain = (; x_lb=-1.0, x_ub=1.0, y_lb=-1.0, y_ub=1.0)
# objects = (PresetGeometries.Rectangle(Lx=0.1, Ly=3, x0=-1, y0=0),)
objects = (PresetGeometries.Circle(R=0.331),)

# Get the flux functions from Trixi
equations = CompressibleEulerEquations2D(1.4)

# Volume fluxes
fs_x(UL, UR) = flux_ranocha(UL, UR, [1.0, 0.0], equations)
fs_y(UL, UR) = flux_ranocha(UL, UR, [0.0, 1.0], equations)

# Boundary flux
# For entropy stability:
fs_boundary(UL, UR, n) = flux_lax_friedrichs(UL, UR, n, equations)

fs = (; fs_x, fs_y, fs_boundary)
const num_fields = 4
const num_dim = 2


# The true solution:
const rho0 = 2.0
const v1_0 = 0.5
const v2_0 = 0.5
const p0   = 3.0

# The true solution:
function trueSoln(x, y, t)
    rho = @. sin(2*pi*(x .-v1_0*t))*sin(2*pi*(y .-v2_0*t)) + rho0

    return prim2cons( SVector{num_fields}([rho, v1_0, v2_0, p0]), equations )
end

# Set the boundary conditions and forcing
function BC(x,y,t, nx, ny, uf)
    # r = sqrt(x^2 + y^2)
    # theta = atan(y, x)
    # if abs(x + 1) < 1e-12 || abs(y + 1) < 1e-12 || 
        # (-pi/4 <= theta <= 3*pi/4 && abs(r-0.331) < 1e-12 )
        return trueSoln(x, y, t)
    # else
    #     return uf
    # end
end

function forcing(u, x,y,t)
    return SVector{4, Float64}(0.0*x, 0.0*x, 0.0*x, 0.0*x)
end

N_deg = 4
cells_per_dimension_x = 32
cells_per_dimension_y = 32
coordinates_min = (domain.x_lb, domain.y_lb)
coordinates_max = (domain.x_ub, domain.y_ub)

rd = RefElemData(Quad(), N=N_deg)
md = MeshData(rd,
            objects,
            cells_per_dimension_x, 
            cells_per_dimension_y;
            coordinates_min=coordinates_min, 
            coordinates_max=coordinates_max,
            precompute_operators=true)

# Apply the initial condition
u = trueSoln.(md.x, md.y, 0.0)
u_float = NamedArrayPartition((; cartesian=unwrap_array(u.cartesian), 
                                cut=unwrap_array(u.cut) ))

# SRD if needed/desired
srd = StateRedistribution(rd, md, eltype(u.cut))

# Generate helper memory and the hybridized SBP operators for use in rhs!
memory = allocate_rhs_memory(md)
cartesian_operators, cut_operators = generate_operators(rd, md)

params = (; cartesian_operators, cut_operators, BC, forcing, fs, md, rd, equations, memory, use_srd=true, use_entropy_vars=false, srd)

# Simulate the PDE
t_end = 1.3
t_span = (0.0, t_end)
t_save = LinRange(t_span..., 130 + 1)

experiment = "test"
prob = ODEProblem(rhs!, u_float, t_span, params)
sol = solve(prob, Tsit5(), adaptive=true, dt=1e-6, abstol=1e-8, reltol=1e-6, saveat=t_save, callback=AliveCallback(alive_interval=20))

# Compute the L2 error
u_true = trueSoln.(md.xq, md.yq, t_end)
u_final_matrix = sol.u[end]
u_final = NamedArrayPartition((; cartesian=wrap_array(u_final_matrix.cartesian, Val{num_fields}()),
                                    cut=wrap_array(u_final_matrix.cut, Val{num_fields}()) ))

# Interpolate u_final to the quadrature points
u_error = similar(u_true)
u_error.cartesian = (u_true.cartesian .- rd.Vq * u_final.cartesian)

Vq_cut = md.mesh_type.cut_cell_operators.volume_interpolation_matrices
for e in axes(u_final.cut,2)
    u_error.cut[:,e] = u_true.cut[:,e] .- Vq_cut[e]*u_final.cut[:,e]
end


include("helperCodes/Plotting_CutMeshes.jl")
x_plot = LinRange(domain.x_lb, domain.x_ub, Integer( ceil(200*(domain.x_ub-domain.x_lb)) ))
y_plot = LinRange(domain.y_lb, domain.y_ub, Integer( ceil(200*(domain.y_ub-domain.y_lb)) ))
V_plot = global_interpolation_op(x_plot, y_plot, u_float, domain, md, rd)


function density(u, x, y, t)
    return u[1,:]
end

function density_error(u, x, y, t)
    u_true = trueSoln.(x,y, t)
    rho_true = getindex.(u_true,1)

    return rho_true - u[1,:]
end

function x_velocity(u, x, y, t)
    return u[2,:] ./ u[1,:]
end

function y_velocity(u, x, y, t)
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
    # return @sprintf("Mach %.3lf, t=%.3lf", v1_prescribed(domain.x_lb, 0.0, t)/c0, t)
    return @sprintf("t=%.2lf", t)
end

plotting_increment = 1;
fps = 20

u_plot = sol.u
makeGIF_grid(u_plot, t_save, density, length(sol.u), md, rd, x_plot, y_plot, V_plot, domain, 
    t_step_incr=plotting_increment,
    plot_lims = (rho0-1, rho0+1),
    filename=@sprintf("figures/entropyWave/eulerEntropyWave_gridPlot_density_%s.gif", experiment),
    fps = fps,
    sol_color=:davos,
    plot_embedded_objects=true,
    line_color=:black,
    titleString=titleString,
)

error_tol = 2e-5
makeGIF_grid(u_plot, t_save, density_error, length(sol.u), md, rd, x_plot, y_plot, V_plot, domain, 
    t_step_incr=plotting_increment,
    plot_lims = (-error_tol, error_tol),
    filename=@sprintf("figures/entropyWave/eulerEntropyWave_gridPlot_densityError_%s.gif", experiment),
    fps = fps,
    sol_color=:broc,
    plot_embedded_objects=true,
    line_color=:black,
    titleString=titleString,
)

using FileIO, JLD2
FileIO.save("savedSoln/euler_entropyWave_L2Error_0.5.jld2", 
    "u_error", u_error,
)


