

using LinearAlgebra
using OrdinaryDiffEq
using RecursiveArrayTools
using SparseArrays
using StaticArrays
using StructArrays

using StartUpDG
using Trixi

const Q_DROP_TOL = 1e-12

function wrap_array(x::Array{T, 3}, ::Val{N}) where {N, T}
    svector_type = SVector{N, eltype(x)}    
    return unsafe_wrap(Matrix{svector_type}, Ptr{svector_type}(pointer(x)), size(x)[2:3])
end

function unwrap_array(x::Matrix{SVector{N, Float64}}) where {N}
    return unsafe_wrap(Array{Float64, 3}, Ptr{Float64}(pointer(x)), (N, size(x, 1), size(x, 2)))
end

function rhs!(du_float, u_float, params, t) 
    @unpack cartesian_operators, cut_operators, BC, forcing, fs, md, rd, equations, memory = params
    @unpack u_tilde, uf_tilde, uP_tilde = memory

    u_cartesian = wrap_array(u_float.cartesian, Val{num_fields}())
    u_cut = wrap_array(u_float.cut, Val{num_fields}())
    u = NamedArrayPartition((; cartesian=u_cartesian, cut=u_cut))
    
    du_cartesian = wrap_array(du_float.cartesian, Val{num_fields}())
    du_cut = wrap_array(du_float.cut, Val{num_fields}())
    du = NamedArrayPartition((; cartesian=du_cartesian, cut=du_cut))

    if params.use_srd == true
        if params.use_entropy_vars == true
            u .= entropy2cons.(params.srd(cons2entropy.(u, equations)))
        else
            params.srd(u)
        end
    end

    # Compute the auxiliary conservative variables that the hSBP operators act on
    if size(u.cartesian,2) != 0
        u_tilde.cartesian .= entropy2cons.(
            cartesian_operators.VhP*cons2entropy.( cartesian_operators.Vq*u.cartesian, equations ), equations)

        nq_cartesian = size(cartesian_operators.Vq,1)
        uf_tilde.cartesian = u_tilde.cartesian[nq_cartesian+1:end,:]
    end
    
    # Note: all cut elements should have the same number of quadrature points
    i_start = size(cut_operators.Vq[1],1) + 1
    Threads.@threads :static for e in axes(u.cut,2)
        u_tilde.cut[e] .= entropy2cons.(
            cut_operators.VhP[e]*cons2entropy.( cut_operators.Vq[e]*u.cut[:,e], equations ), equations )

        fids = md.mesh_type.cut_face_nodes[e]
        i_end = length(u_tilde.cut[e])
        uf_tilde.cut[fids] = u_tilde.cut[e][ i_start:i_end ]
    end

    # Initialize uP_tilde
    Threads.@threads :static for id in eachindex(md.mapP)
        uP_tilde[id] = uf_tilde[md.mapP[id]]
    end

    # Apply BCs
    Threads.@threads :static for i in eachindex(md.mapP)
        if i == md.mapP[i]
            uP_tilde[i] = BC(md.xf[i], md.yf[i], t, md.nx[i], md.ny[i], uf_tilde[i])
        end
    end

    if size(u_tilde.cartesian,2) != 0
        rhs_cartesian!(du.cartesian, 
                    u.cartesian,
                    u_tilde.cartesian, 
                    uf_tilde.cartesian,
                    uP_tilde.cartesian, 
                    (; cartesian_operators, forcing, fs, md, rd, equations), 
                    t)
    end

    rhs_cut!(du.cut, 
              u.cut,
              u_tilde.cut,
             uf_tilde.cut,
             uP_tilde.cut,
             (; cut_operators, forcing, fs, md, equations),
             t)

    
    # if params.use_srd == true
    #     params.srd(du)
    # end
end


function rhs_cartesian!(du, u, u_tilde, uf_tilde, uP_tilde, params, t)
    @unpack cartesian_operators, forcing, fs, md, rd, equations = params
    @unpack fs_x, fs_y, fs_boundary = fs

    nJ   = cartesian_operators.nJ
    Ph   = cartesian_operators.Ph
    LIFT = cartesian_operators.LIFT
    IJV_Qrsh = cartesian_operators.IJV_Qrsh

    # Compute the boundary terms
    du .= LIFT*fs_boundary.(uf_tilde, uP_tilde, nJ)
    
    # Compute the volume terms
    du_volume = similar(u_tilde)
    fill!(du_volume, zero(eltype(du_volume)))
    Threads.@threads :static for e in axes(u_tilde, 2) # for each element
        for k in eachindex(IJV_Qrsh[1][1])
            i, j = IJV_Qrsh[1][1][k], IJV_Qrsh[1][2][k]

            QxFx_ij = md.rxJ.cartesian[1,1] * IJV_Qrsh[1][3][k]*fs_x(u_tilde[i,e], u_tilde[j,e])
            du_volume[i,e] +=  QxFx_ij
            du_volume[j,e] += -QxFx_ij
        end

        for k in eachindex(IJV_Qrsh[2][1])
            i, j = IJV_Qrsh[2][1][k], IJV_Qrsh[2][2][k]

            QyFy_ij = md.syJ.cartesian[1,1] * IJV_Qrsh[2][3][k]*fs_y(u_tilde[i,e], u_tilde[j,e])
            du_volume[i,e] +=  QyFy_ij
            du_volume[j,e] += -QyFy_ij
        end
    end
    
    du .+= 2*Ph*du_volume
    du .= -du ./ md.J.cartesian + forcing.(u, md.x.cartesian, md.y.cartesian, t)
end

# uf = nodes on faces
# uP = ghost nodes on boundaries
function rhs_cut!(du, u, u_tilde, uf_tilde, uP_tilde, params, t)
    @unpack cut_operators, forcing, fs, md, equations = params

    inv_M    = cut_operators.inv_M
    Vf   = cut_operators.Vf
    Vh   = cut_operators.Vh
    wf   = cut_operators.wf
    nJ   = cut_operators.nJ
    Qxyh = cut_operators.Qxyh

    fill!(du, zero(eltype(du)))
    Threads.@threads :static for e in eachindex(u_tilde) # for each element
        fids = md.mesh_type.cut_face_nodes[e]

        # Compute the boundary terms
        du_boundary_e = Vf[e]'*(wf[fids] .*fs_boundary.(uf_tilde[fids], uP_tilde[fids], nJ[fids]))

        # Compute the volume terms
        du_volume_e = similar.(u_tilde[e])
        fill!(du_volume_e, zero(eltype(u_tilde[e])))
        for d in eachindex(Qxyh[e])
            I_e, J_e, Q_ed = findnz(Qxyh[e][d])

            for k in eachindex(I_e)
                i, j = I_e[k], J_e[k]

                QdFd_ij = Q_ed[k]*fs[d](u_tilde[e][i], u_tilde[e][j])
                du_volume_e[i] +=  QdFd_ij
                du_volume_e[j] += -QdFd_ij
            end
        end
        
        du[:,e] .= inv_M[e]*(du_boundary_e + 2*Vh[e]'*du_volume_e)
    end
    
    du .= -du + forcing.(u, md.x.cut, md.y.cut, t)
end


function compute_entropy_residual(u_matrix, M, t, params, equations)
    u_cartesian = wrap_array(u_matrix.cartesian, Val{num_fields}())
    u_cut = wrap_array(u_matrix.cut, Val{num_fields}())
    u = NamedArrayPartition((; cartesian=u_cartesian, cut=u_cut))
    
    v = similar(u)
    v.cartesian .= params.rd.Pq*cons2entropy.(params.cartesian_operators.Vq*u.cartesian, equations)
    for e in axes(u.cut, 2)
        v.cut[:,e] .= ( M.cut[e] \ params.cut_operators.Vq[e]'*diagm(params.md.wJq.cut[:,e]) )*
            cons2entropy.(params.cut_operators.Vq[e]*u.cut[:,e], equations)
    end

    du_matrix = similar(u_matrix)
    rhs!(du_matrix, u_matrix, params, t)
    
    du_cartesian = wrap_array(du_matrix.cartesian, Val{num_fields}())
    du_cut = wrap_array(du_matrix.cut, Val{num_fields}())
    du = NamedArrayPartition((; cartesian=du_cartesian, cut=du_cut))

    duM = similar(u)
    duM.cartesian .= params.md.J.cartesian .* (M.cartesian * du.cartesian)
    for e in axes(du.cut, 2)
        duM.cut[:,e] .= M.cut[e] * du.cut[:,e]
    end

    return sum(dot.(v, duM))
end


function allocate_rhs_memory(md)
    # Allocate memory for the auxiliary conservative variables
    num_tilde_vars = size(md.xq.cartesian,1) + size(md.xf.cartesian,1)
    u_tilde_cartesian = zeros(SVector{num_fields, Float64}, ( num_tilde_vars, size(md.x.cartesian,2) ) )
    
    u_tilde_cut = Vector{SVector{num_fields, Float64}}[]
    nq_cut =  size(md.xq.cut,1)
    for e in 1:size(md.x.cut,2)
        fids = md.mesh_type.cut_face_nodes[e]
        num_tilde_vars = nq_cut + length(md.xf.cut[fids])
        u_tilde_cut_e = zeros(SVector{4, Float64}, num_tilde_vars)
        push!(u_tilde_cut, u_tilde_cut_e)
    end
    u_tilde = (; cartesian=u_tilde_cartesian, cut=u_tilde_cut)

    # Allocate memory for face values
    uf_tilde_cartesian = StructArray{SVector{num_fields, Float64}}(ntuple(_ -> similar(Float64.(md.mapP.cartesian)), num_fields))
    uf_tilde_cut = StructArray{SVector{num_fields, Float64}}(ntuple(_-> similar(Float64.(md.mapP.cut)), num_fields))

    uf_tilde = NamedArrayPartition((; cartesian=uf_tilde_cartesian, cut=uf_tilde_cut))
    uP_tilde = similar(uf_tilde)

    return (; u_tilde, uf_tilde, uP_tilde)
end

function generate_operators(rd, md)
    # Compute hybridized SBP operators for the reference element
    Qrsh, VhP, Ph, _ = hybridized_SBP_operators(rd)

    # To speed up computation and reduce memory, only save the lower-triangular portion of the Q's,
    # excluding the diagonal
    Qrsh_tri = tril.(Qrsh, -1)
    Qrsh_tri_sparse = sparse.(Qrsh_tri)
    droptol!.(Qrsh_tri_sparse, Q_DROP_TOL)
    IJV_Qrsh = findnz.(Qrsh_tri_sparse)

    cartesian_operators = (; Qrsh=Qrsh_tri_sparse, 
                IJV_Qrsh=IJV_Qrsh,
                Ph, 
                VhP=VhP,
                M=rd.M, 
                Vf=rd.Vf, 
                Vq=rd.Vq,
                LIFT=rd.LIFT,
                nJ=StructArray{SVector{num_dim, Float64}}((md.nxJ.cartesian, md.nyJ.cartesian)))

    # Compute hybridized SBP operators for the cut elements
    Qxyh, VhP, _, Vh = hybridized_SBP_operators(md)

    # To save computation and memory only save the strictly lower triangular portion of the
    # Q's. Note I,J can be different for each cut element since cut elements can have an
    # arbitrary number of faces, so we do not pre-compute them.
    Qxyh_tri_sparse = Vector{typeof(Qrsh_tri_sparse)}()
    for e in eachindex(Qxyh)
        Qxyh_e_sparse = sparse.(tril.(Qxyh[e], -1))
        droptol!.(Qxyh_e_sparse, Q_DROP_TOL)

        push!(Qxyh_tri_sparse, Qxyh_e_sparse)
    end

    (; wJf) = md.mesh_type.cut_cell_data
    wf = wJf ./ md.Jf
    cut_operators = (; Qxyh=Qxyh_tri_sparse,
                Vh, 
                VhP=VhP,
                M=md.mesh_type.cut_cell_operators.mass_matrices,
                inv_M=inv.(md.mesh_type.cut_cell_operators.mass_matrices), 
                Vf=md.mesh_type.cut_cell_operators.face_interpolation_matrices,
                Vq=md.mesh_type.cut_cell_operators.volume_interpolation_matrices,
                nJ=StructArray{SVector{num_dim, Float64}}((md.nxJ.cut, md.nyJ.cut)),
                wf=wf.cut)

    return cartesian_operators, cut_operators
end