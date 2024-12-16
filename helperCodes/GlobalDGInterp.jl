# Generate a global interpolation operator for generating u values on a grid defined
# by x_desired, y_desired. Values within the desired grid falling in the embedded
# objects or outside the embedding domain will be assigned "outside_value."
#
# Notes:
# - If an point in the desired grid is within an element, it depends only on the element
# - For points on the boundary, take the average of the 2 elements who share the boundary
# - The final global interpolation operator is generated 
function global_interpolation_op(x_desired, y_desired, u, domain, md, rd)
    nx_desired = length(x_desired)
    ny_desired = length(y_desired)
    n_desired = nx_desired*ny_desired
 
    nx = md.mesh_type.cut_cell_data.cells_per_dimension[1]
    ny = md.mesh_type.cut_cell_data.cells_per_dimension[2]

    x_min_global, y_min_global = domain.x_lb, domain.y_lb
    hx, hy = (domain.x_ub - domain.x_lb) / nx, (domain.y_ub - domain.y_lb) / ny
    
    V_cartesian = Matrix{Float64}[]
    V_cut = Matrix{Float64}[]

    i_cartesian = Vector{Vector{Int64}}(undef, size(u.cartesian, 3))
    i_cut = Vector{Vector{Int64}}(undef, size(u.cut, 3))
    ix_all = 1:nx_desired
    iy_all = 1:ny_desired
    i_all = 1:n_desired

    count = zeros(Int64, n_desired)
    for e in axes(u.cartesian,3)
        i_cartesian[e] = Int64[]

        # Find the desired points within this element
        x_min, x_max = minimum(md.xf.cartesian[:,e]), maximum(md.xf.cartesian[:,e])
        y_min, y_max = minimum(md.yf.cartesian[:,e]), maximum(md.yf.cartesian[:,e])

        ix_contained = ix_all[ (x_desired .>= x_min) .&& (x_desired .<= x_max) ]
        iy_contained = iy_all[ (y_desired .>= y_min) .&& (y_desired .<= y_max) ]

        # Create a vector of all the points in this element
        x_contained = zeros(eltype(md.x), length(ix_contained)*length(iy_contained))
        y_contained = zeros(eltype(md.x), length(ix_contained)*length(iy_contained))
        nx_local = length(ix_contained)
        for iy in eachindex(iy_contained)
            for ix in eachindex(ix_contained)
                x_contained[ix + (iy-1)*nx_local] = x_desired[ix_contained[ix]]
                y_contained[ix + (iy-1)*nx_local] = y_desired[iy_contained[iy]]

                i_global_desired = ix_contained[ix] + (iy_contained[iy]-1)*nx_desired
                push!(i_cartesian[e], i_global_desired)

                # Increment the visit count for contained points
                count[i_global_desired] += 1
            end
        end

        # Construct the local interpolation matrix
        if !isempty(x_contained)
            # map (x_contained, y_contained) to [-1,1]^2
            x_contained_ref = 2*(x_contained .- x_min) ./ hx .- 1.0
            y_contained_ref = 2*(y_contained .- y_min) ./ hy .- 1.0
            V_e = vandermonde(Quad(), rd.N, x_contained_ref, y_contained_ref) / rd.VDM
        else
            V_e = zeros(eltype(md.x), (0,size(u.cartesian,2)))
        end
        push!(V_cartesian, V_e)
    end

    # Repeat process for cut elements
    for e in axes(u.cut,3)
        i_cut[e] = Int64[]

        # Find the desired points within the CARTESIAN BACKGROUND element
        I = md.mesh_type.cut_cell_data.linear_to_cartesian_element_indices.cut[e]
        Ix, Iy = I[1], I[2]

        x_min, x_max = x_min_global + (Ix-1)*hx, x_min_global + Ix*hx
        y_min, y_max = y_min_global + (Iy-1)*hy, y_min_global + Iy*hy

        ix_contained_cart = ix_all[ (x_desired .>= x_min) .&& (x_desired .<= x_max) ]
        iy_contained_cart = iy_all[ (y_desired .>= y_min) .&& (y_desired .<= y_max) ]

        x_contained = Float64[]
        y_contained = Float64[]
        for iy in eachindex(iy_contained_cart)
            for ix in eachindex(ix_contained_cart)
                x, y = x_desired[ix_contained_cart[ix]], y_desired[iy_contained_cart[iy]]

                # See if the point is within the actual cut cell and not just its background
                if is_contained(md.mesh_type.cut_cell_data.cutcells[e], [x,y])
                    # i_global_desired = (ix_contained_cart[ix]-1)*ny_desired + iy_contained_cart[iy]
                    i_global_desired = ix_contained_cart[ix] + (iy_contained_cart[iy]-1)*nx_desired
                    push!(i_cut[e], i_global_desired)

                    push!(x_contained, x)
                    push!(y_contained, y)

                    count[i_global_desired] += 1
                end
            end
        end

        # Construct the element-local interpolation Matrix
        if !isempty(x_contained)
            elem = md.mesh_type.physical_frame_elements[e]
            VDM = vandermonde(elem, rd.N, md.x.cut[:, e], md.y.cut[:, e])
            V_e = vandermonde(elem, rd.N, x_contained, y_contained) / VDM
        else
            V_e = zeros(eltype(md.x), (0,size(u.cut, 2)))
        end
        push!(V_cut, V_e)
    end

    # Adjust points that multiple elements contributed to
    for e in axes(u.cartesian,3)
        for i in eachindex(i_cartesian[e])
            if count[i_cartesian[e][i]] > 1
                V_cartesian[e][i,:] ./= count[i_cartesian[e][i]]
            end
        end
    end

    for e in axes(u.cut,3)
        for i in eachindex(i_cut[e])
            if count[i_cut[e][i]] > 1
                V_cut[e][i,:] ./= count[i_cut[e][i]]
            end
        end
    end

    # Contruct the final object
    V_global = (; size=(nx_desired, ny_desired),
        cartesian=V_cartesian,
        cut=V_cut,
        indices=(;cartesian=i_cartesian, cut=i_cut),
        i_outside=i_all[count .== 0],
        counts=count
    )
    return V_global
end

function apply_global_interp!(u_desired, u, x, y, t, func, V_global; outside_val=0.0)
    fill!(u_desired, zero(eltype(u_desired)))
    for e in axes(u.cartesian,3)
        u_desired[V_global.indices.cartesian[e]] += V_global.cartesian[e]*func(u.cartesian[:, :, e], x.cartesian[:,e], y.cartesian[:,e], t)
    end

    for e in axes(u.cut,3)
        u_desired[V_global.indices.cut[e]] += V_global.cut[e]*func(u.cut[:, :, e], x.cut[:,e], y.cut[:,e], t)
    end

    if outside_val != 0
        u_desired[V_global.i_outside] .= outside_val
    end
end

function apply_global_interp(u, x, y, t, func, V_global; outside_val=0.0)
    u_desired = zeros(eltype(u), V_global.size)
    apply_global_interp!(u_desired, u, x, y, t, func, V_global, outside_val=outside_val)

    return u_desired
end