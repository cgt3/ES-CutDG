# SWE 2D solver
using LinearAlgebra
using OrdinaryDiffEq
using Plots
using RecursiveArrayTools
using StaticArrays
using StructArrays

using PathIntersections
using StartUpDG
using Trixi

include("alive.jl")
include("es_rhs.jl")

# Simulation parameters

t_end = 1.0
t_span = (0.0, t_end)
t_save = LinRange(t_span..., 100 + 1)

domain = (; x_lb=-1.0, x_ub=1.0, y_lb=-1.0, y_ub=1.0)
# objects = (PresetGeometries.Rectangle(Lx=0.1, Ly=3, x0=-1, y0=0),)
objects = (PresetGeometries.Circle(R=0.331),)

# Get the flux functions from Trixi
equations = ShallowWaterEquations2D(gravity_constant=1.0)

# Volume fluxes
fs_x(UL, UR) = flux_wintermeyer_etal(UL, UR, 1, equations)
fs_y(UL, UR) = flux_wintermeyer_etal(UL, UR, 2, equations)

# Boundary flux
# For entropy conservation:
# fs_boundary(UL, UR, n) = flux_wintermeyer_etal(UL, UR, n, equations)

# For entropy stability:
fs_boundary(UL, UR, n) = flux_lax_friedrichs(UL, UR, n, equations)

# or
# fs_LaxFriedrichs = FluxPlusDissipation(flux_wintermeyer_etal, DissipationLocalLaxFriedrichs())
# fs_boundary(UL, UR, n) = fs_LaxFriedrichs(UL, UR, n, equations)

fs = (; fs_x, fs_y, fs_boundary)
const num_fields = 4
const num_dim = 2


# Construct the DG operators using StartUpDG
N_deg = 4;
cells_per_dimension_x = 16
cells_per_dimension_y = 16
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

# Set the initial condition
function u_allocate(x,y)
    h = 0.0
    v1 = 0.0
    v2 = 0.0
    b = 0.0

    return SVector{num_fields}([h, h*v1, h*v2, b])
end


function uIC!(u, md)
    function initialize_elements!(u, x, y)
        num_nodes = size(u,1)
        for e in axes(u,2)
            x_avg, y_avg = sum(x[:,e]) / num_nodes, sum(y[:,e]) / num_nodes

            if y_avg > domain.y_ub - 0.5
                for i in axes(u,1)
                    u[i,e] = SVector{num_fields, Float64}(4.0, 0.0, 0.0, 0.0)
                end
            else
                for i in axes(u,1)
                    u[i,e] = SVector{num_fields, Float64}(3.0, 0.0, 0.0, 0.0)
                end
            end
        end
    end

    # Process the Cartesian elements
    initialize_elements!(u.cartesian, md.x.cartesian, md.y.cartesian)

    # Process the cut elements
    initialize_elements!(u.cut, md.x.cut, md.y.cut)
end

# Set the boundary conditions and forcing
function BC(x,y,t, nx, ny, uf)
    v_n = uf[2]*nx + uf[3]*ny
    return SVector{4, Float64}(uf[1], uf[2] - 2*v_n*nx, uf[3] - 2*v_n*ny, uf[4])

    # return SVector{4, Float64}(uf[1], -uf[2], -uf[3], uf[4])
end

function forcing(u,x,y,t)
    return SVector{4, Float64}(0.0*x, 0.0*x, 0.0*x, 0.0*x)
end

# Apply the initial condition
u = NamedArrayPartition((; cartesian=u_allocate.(md.x.cartesian, md.y.cartesian), cut=u_allocate.(md.x.cut, md.y.cut)))
uIC!(u, md)

u_float = NamedArrayPartition((; cartesian=unwrap_array(u.cartesian), cut=unwrap_array(u.cut)))

# # SRD if needed/desired
srd = StateRedistribution(rd, md, eltype(u.cut))

# Generate helper memory and the hybridized SBP operators for use in rhs!
memory = allocate_rhs_memory(md)
cartesian_operators, cut_operators = generate_operators(rd, md)


# Simulate the PDE
params_noSRD = (; cartesian_operators, cut_operators, BC, forcing, fs, md, rd, equations, memory, use_srd=false, use_entropy_vars=false, srd)
prob = ODEProblem(rhs!, u_float, t_span, params)
sol_noSRD = solve(prob, Tsit5(), dt=1e-4, saveat=t_save, callback=AliveCallback(alive_interval=50))

params_uSRD = (; cartesian_operators, cut_operators, BC, forcing, fs, md, rd, equations, memory, use_srd=true, use_entropy_vars=false, srd)
prob = ODEProblem(rhs!, u_float, t_span, params)
sol_uSRD = solve(prob, Tsit5(), dt=1e-4, saveat=t_save, callback=AliveCallback(alive_interval=50))

params_vSRD = (; cartesian_operators, cut_operators, BC, forcing, fs, md, rd, equations, memory, use_srd=true, use_entropy_vars=true, srd)
prob = ODEProblem(rhs!, u_float, t_span, params)
sol_vSRD = solve(prob, Tsit5(), dt=1e-4, saveat=t_save, callback=AliveCallback(alive_interval=50))

# Calculate the entropy residual
M = (; cartesian=cartesian_operators.M, cut=cut_operators.M)
entropy_res_noSRD = zeros(Float64, length(sol_noSRD.u))
entropy_res_uSRD = zeros(Float64, length(sol_SRD.u))
entropy_res_vSRD = zeros(Float64, length(sol_SRD.u))
for k in eachindex(t_save)
    entropy_res_noSRD[k] = compute_entropy_residual(sol_noSRD.u[k], M, t_save[k], params_noSRD, equations)
    entropy_res_uSRD[k] = compute_entropy_residual(sol_uSRD.u[k], M, t_save[k], params_SRD, equations)
    entropy_res_vSRD[k] = compute_entropy_residual(sol_vSRD.u[k], M, t_save[k], params_SRD, equations)
end

# Plot the entropy residual
entropy_res_plot = Plots.plot(t_save, [entropy_res_noSRD entropy_res_uSRD entropy_res_vSRD],
    labels=["No SRD" "SRD(U)" "SRD(V)"],
    linewidth=2,
    xlabel="t",
    xguidefontsize=16,
    xtickfontsize=10,
    ytickfontsize=10)
display(entropy_res_plot)
# png("figures/entropyRes_ES_noScaling.png")

entropy_res_plot = Plots.plot(t_save, [entropy_res_noSRD entropy_res_SRD entropy_res_vSRD], 
    ylims=(-0.2, 0.0),
    labels=["No SRD" "SRD(U)" "SRD(V)"],
    linewidth=2,
    xlabel="t",
    xguidefontsize=16,
    xtickfontsize=10,
    ytickfontsize=10)
display(entropy_res_plot)
# png("figures/entropyRes_ES_scaled.png")


