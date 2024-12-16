# SWE 2D solver
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

t_end = 0.3
t_span = (0.0, t_end)
t_save = LinRange(t_span..., 1 + 1)

domain = (; x_lb=-1.0, x_ub=1.0, y_lb=-1.0, y_ub=1.0)
# objects = (PresetGeometries.Rectangle(Lx=0.1, Ly=3, x0=-1, y0=0),)
objects = (PresetGeometries.Circle(R=0.331),)

# Get the flux functions from Trixi
equations = ShallowWaterEquations2D(gravity_constant=1.0)

# Volume fluxes
fs_x(UL, UR) = flux_wintermeyer_etal(UL, UR, 1, equations)
fs_y(UL, UR) = flux_wintermeyer_etal(UL, UR, 2, equations)

# For entropy stability:
fs_boundary(UL, UR, n) = flux_lax_friedrichs(UL, UR, n, equations)

fs = (; fs_x, fs_y, fs_boundary)
const num_fields = 4
const num_dim = 2

# The true solution:
function trueSoln(x, y, t)
    h = sin(2*pi*x)*sin(2*pi*y)*cos(pi*t) + 3
    return SVector{num_fields}([h, h, h, 0*h ])
end

# Set the boundary conditions and forcing
function BC(x,y,t, nx, ny, uf)
    return trueSoln(x, y, t)
end

function forcing(u, x,y,t)
    cx, sx = cos(2*pi*x), sin(2*pi*x)
    cy, sy = cos(2*pi*y), sin(2*pi*y)
    ct, st = cos(pi*t), sin(pi*t)

    return SVector{4, Float64}(
        -pi*sx*sy*st + 2*pi*(cx*sy + sx*cy)*ct,
        -pi*sx*sy*st + 2*pi*(cx*sy + sx*cy)*ct + 0.5*(4*pi*cx*sx*sy*sy*ct*ct + 12*pi*cx*sy*ct),
        -pi*sx*sy*st + 2*pi*(cx*sy + sx*cy)*ct + 0.5*(4*pi*sx*sx*cy*sy*ct*ct + 12*pi*sx*cy*ct),
        0.0*x)
end

L2_error = zeros(Float64, (7,5))
for i_mesh = 2:5
        if i_mesh > 3
            adaptive = false
            dt_start=5e-4
        else
            adaptive = true
            dt_start= 1e-4
        end
    for N_deg in [2, 3, 4]
        cells_per_dimension_x = 2^i_mesh
        cells_per_dimension_y = 2^i_mesh
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
        prob = ODEProblem(rhs!, u_float, t_span, params)
        sol = solve(prob, Tsit5(), adaptive=adaptive, dt=dt_start, saveat=t_save, callback=AliveCallback(alive_interval=50))

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

        # Apply the quadrature weights and sum
        L2_error[i_mesh, N_deg] = sum(dot.(u_error, md.wJq .* u_error))
        println("\n\n Finished i_mesh=$i_mesh, N_deg=$N_deg \n\n")
    end
end

using FileIO, JLD2
FileIO.save("savedSoln/swe_MMS_L2Error_0.3.jld2", 
    "L2_error", L2_error,
)


