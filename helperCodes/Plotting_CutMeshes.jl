
using Plots
using GLMakie

using Colors
using ColorSchemes

using Triangulate
using TriplotBase
using TriplotRecipes
using TriplotRecipes: TriPseudocolor

include("GlobalDGInterp.jl")

## Good color maps for SWE:
# color=:oslo
# color=:grayyellow,
# color=:lapaz, # oslo but w pinkish hue on white end
# color=:davos, # oslo but w greenish-yellow hue on white end
# color=:devon, # oslo but w purple hue on white end
# color=:cividis, # blue-grey-yellow
# color=:PuBu, # pinkish-white to (green) blue
# color=:GnBu, # greenish-yellow to dark blue
# color=:Blues # white to dark blue
# color=:atlantic, # dark grey to blue-green-grey to purple-grey

function build_triangulations(md, objects)
    # Build the triangulation for Cartesian elements
    if size(md.xf.cartesian,2) != 0
        triin_cartesian=Triangulate.TriangulateIO()
        triin_cartesian.pointlist = hcat( vcat(md.xf.cartesian[:,1], md.x.cartesian[:,1]),
                                          vcat(md.yf.cartesian[:,1], md.y.cartesian[:,1]) )'
        triout_cartesian, _ = triangulate("Q", triin_cartesian)
        t_cartesian = triout_cartesian.trianglelist
    end
    
    # TODO: cut cells do not contain enough points on their boundary to create a
    # good plot; in particular they also do not contain their corner points
    
    # Build the triangulations for the cut cells
    t_cut = Matrix{Int32}[]
    Vc = Matrix{Float64}[]
    xyc = Matrix{Float64}[]
    for e=1:size(md.x.cut,2) 
        fids = md.mesh_type.cut_face_nodes[e]

        stop_pts = md.mesh_type.cut_cell_data.cutcells[e].stop_pts
        xy_stop_pts = md.mesh_type.cut_cell_data.cutcells[e].(stop_pts)

        xc = getindex.(xy_stop_pts,1)
        yc = getindex.(xy_stop_pts,2)

        elem = md.mesh_type.physical_frame_elements[e]
        VDM = vandermonde(elem, rd.N, md.x.cut[:, e], md.y.cut[:, e])
        Vc_e  = vandermonde(elem, rd.N, xc, yc) / VDM
        push!(Vc, Vc_e)
        push!(xyc, [xc yc])

        triin_cut=Triangulate.TriangulateIO()
        triin_cut.pointlist = hcat( vcat(md.xf.cut[fids], xc, md.x.cut[:,e]),
                                    vcat(md.yf.cut[fids], yc, md.y.cut[:,e]) )'
        triout_cut, _ = triangulate("Q", triin_cut)
    
        t_cut_e = triout_cut.trianglelist
        push!(t_cut, t_cut_e)
    end
    
    # Build the triangulations for the embedded objects to plot over
    # since the boundary nodes do not contain cell corners and can 
    # consequently cause issues
    plist_objects = TriPseudocolor[]
    s = 0:0.01:1
    for i in eachindex(objects)
        pts = objects[i].(s)
        x_obj, y_obj = getindex.(pts,1), getindex.(pts,2)
    
        triin_cut=Triangulate.TriangulateIO()
        triin_cut.pointlist = hcat( x_obj, y_obj )'
    
        # Set the boundary nodes
        n_pts = length(x_obj)
        triin_cut.segmentlist = zeros(Int32, (2, n_pts));
        triin_cut.segmentlist[1,:] = 1:n_pts
        triin_cut.segmentlist[2,:] = mod.(1:n_pts, n_pts) .+ 1
        triout_cut, _ = triangulate("pQ", triin_cut)
        t_obj_i = triout_cut.trianglelist
    
        push!(plist_objects, TriPseudocolor( x_obj, y_obj, 
                                            -1000*ones(length(x_obj)), t_obj_i) )
    
    end

    return (; t_cartesian, t_cut, Vc, xyc, plist_objects)
end


function makeGIF_tri(u, t_save, func, n_t, md, rd, triangulations, domain; 
                       plot_embedded_objects=false, 
                       t_step_incr=1, 
                       sol_color=:oslo,
                       plot_lims=(0.0, 1.0),
                       fps=25,
                       filename="makeGif.gif",
                       titleString=(f(t) = "") 
        )

    @unpack t_cartesian, t_cut, Vc, xyc, plist_objects = triangulations

    Vf = md.mesh_type.cut_cell_operators.face_interpolation_matrices

    anim = @animate for k=1:t_step_incr:n_t
        plist = TriPseudocolor[]
        # Process the Cartesian elements
        for e=1:size(md.x.cartesian,2)
            u_e = func(u[k].cartesian[:, :, e], md.x.cartesian[:,e], md.y.cartesian[:,e], t_save[k])
            uf_e = rd.Vf * u_e
            push!(plist, TriPseudocolor( vcat(md.xf.cartesian[:,e], md.x.cartesian[:,e]),
                                         vcat(md.yf.cartesian[:,e], md.y.cartesian[:,e]),
                                         vcat(uf_e, u_e), t_cartesian ) )
        end
    
        # Process the cut elements
        for e=1:size(md.x.cut,2)
            fids = md.mesh_type.cut_face_nodes[e]
    
            u_e = func(u[k].cut[:, :, e], md.x.cut[:,e], md.y.cut[:,e], t_save[k])
            uf_e = Vf[e] * u_e
            uc_e = Vc[e] * u_e

            push!(plist, TriPseudocolor( vcat(md.xf.cut[fids], xyc[e][:,1], md.x.cut[:,e]),
                                         vcat(md.yf.cut[fids], xyc[e][:,2], md.y.cut[:,e]),
                                         vcat(uf_e, uc_e, u_e), t_cut[e] ) )
        end
    
        # Plot the solution
        Plots.plot(plist, ylims=(domain.y_lb, domain.y_ub), 
            xlims=(domain.x_lb, domain.x_ub), 
            color=sol_color,
            clims=plot_lims, 
            colorbar=true,
            rightmargin=10Plots.mm,
            ratio=1)
        
        # Plot the embedded objects
        if plot_embedded_objects == true
            Plots.plot!(plist_objects, 
                ylims=(domain.y_lb, domain.y_ub), 
                xlims=(domain.x_lb, domain.x_ub), 
                clims=plot_lims, 
                color=:grays, 
                colorbar=true)
                
            Plots.surface!([domain.x_lb], [domain.y_lb], [0],
            ylims=(domain.y_lb, domain.y_ub), 
            xlims=(domain.x_lb, domain.x_ub), 
            clims=plot_lims, 
            color=sol_color,
            leg=false,
            colorbar=true)
        end

        Plots.plot!(title=titleString(t_save[k]))
    end
    gif(anim, filename, fps=fps)
end


function makeGIF_grid(u, t_save, func, n_t, md, rd, x_plot, y_plot, V_plot, domain; 
        plot_embedded_objects=false, 
        t_step_incr=1, 
        sol_color=:oslo,
        plot_lims=(0.0, 1.0),
        fps=25,
        outside_val = 0,
        filename="makeGif.gif",
        line_color =:black,
        titleString=(f(t) = ""),
    )

    s = 0.0:0.01:1.0
    x_obj = Vector{Float64}[]
    y_obj = Vector{Float64}[]
    for object in objects
        pts = object.(s)
        push!(x_obj, getindex.(pts,1))
        push!(y_obj, getindex.(pts,2))
    end
    
    anim = @animate for k=1:t_step_incr:n_t
        z_vals = apply_global_interp(u[k], md.x, md.y, t_save[k], func, V_plot, outside_val=outside_val)

        # Plot the solution
        Plots.heatmap(x_plot, y_plot, z_vals',
            ylims=(domain.y_lb, domain.y_ub), 
            xlims=(domain.x_lb, domain.x_ub), 
            color=sol_color,
            clims=plot_lims, 
            colorbar=true,
            aspect_ratio=1, #(domain.y_ub - domain.y_lb)/(domain.x_ub - domain.x_lb),
            rightmargin=10Plots.mm
        )

        # Plot the embedded objects
        if plot_embedded_objects == true
            for i_object in eachindex(objects)
                Plots.plot!(x_obj[i_object], y_obj[i_object],
                    color=line_color,
                    linewidth=1.0, leg=false)
            end
        end

        Plots.plot!(title=titleString(t_save[k]))
    end
    gif(anim, filename, fps=fps)
end


function makeMovie_GLMakie(u, x, y, t, func, objects, n_t, x_plot, y_plot, V_plot;
        t_step_incr=1, 
        sol_color=:oslo,
        filename="test.mp4",
        aspect=(1.0, 1.0, 0.67),
        obj_val=1.0,
        outside_val=NaN,
        zlims=(0.0, 1.0),
        fps = 26,
        clims=zlims,
        azimuth = -3*pi/4,
        elevation = pi/4,
        perspectiveness=0.25
     )

    s = 0.0:0.01:1.0
    h = 0.0:0.1:obj_val
    x_obj = Matrix{Float64}[]
    y_obj = Matrix{Float64}[]
    z_obj = Matrix{Float64}[]
    num_pts_h = length(h)
    num_pts_s = length(s)
    for object in objects
        pts = object.(s)
        push!(x_obj, ones(num_pts_h)*getindex.(pts,1)')
        push!(y_obj, ones(num_pts_h)*getindex.(pts,2)')
        push!(z_obj, h*ones(num_pts_s)')
    end

    function obj_plot(k)
        return z_obj
    end
    
    function u_plot(u, func, V_plot, k)
        return apply_global_interp(u[k], x, y, t[k], func, V_plot, outside_val=outside_val)
    end
    
    i_time = Observable(1)
    z_plot = @lift(u_plot(u, func, V_plot, $i_time))
    # z_obj_obs = @lift(obj_plot($i_time))

    fig = GLMakie.Figure()
    ax = GLMakie.Axis3(fig[1,1])
    ax.azimuth = azimuth
    ax.elevation = elevation
    ax.perspectiveness=perspectiveness
    ax.aspect= aspect

    for e in eachindex(objects)
        GLMakie.surface!(x_obj[e], y_obj[e], z_obj[e], colormap=:grays, colorrange=(-1.0, 0.0))
    end
    GLMakie.surface!(x_plot, y_plot, z_plot, colormap=sol_color, colorrange=clims)

    # Generate the movie
    timestamps=range(1, n_t, step=t_step_incr)
    framerate = fps
    record(fig, filename, timestamps; framerate=framerate) do k
        i_time[ ] = k
    end    
end
