#=
WilsonFermion which has extra implementations to speed up computations with WilsonFermion.jl.
=#
module ParWilsonFermionModule

    using LinearAlgebra
    using Base.Threads
    using StaticArrays

    # have no idea why using doesn't work but everything breaks
    # using LatticeQCD.Actions:FermiActionParam,FermiActionParam_Wilson
    # using LatticeQCD.AbstractFermion:FermionFields,Wx!,Wdagx!,clear!,substitute_fermion!,Dx!,fermion_shift!,fermion_shiftB!,add!,set_wing_fermi!,WdagWx!,apply_periodicity,Ddagx!
    # using LatticeQCD.WilsonFermion_module:WilsonFermion,Wx!,Wdagx!,Dx!,Ddagx!,mk_gamma
    #
    # import LatticeQCD.AbstractFermion:substitute_fermion!

    import LatticeQCD.WilsonFermion_module:WilsonFermion,Wx!,Wdagx!,Dx!,Ddagx!,mk_gamma
    import LatticeQCD.Actions:FermiActionParam,FermiActionParam_Wilson,
                FermiActionParam_WilsonClover,FermiActionParam_Staggered
    import LatticeQCD.Gaugefields:GaugeFields,GaugeFields_1d,SU3GaugeFields,SU2GaugeFields,SU3GaugeFields_1d,SU2GaugeFields_1d,
                staggered_phase,SUn,SU2,SU3,SUNGaugeFields,SUNGaugeFields_1d,SU
    import LatticeQCD.AbstractFermion:FermionFields,
        Wx!,Wdagx!,clear!,substitute_fermion!,Dx!,fermion_shift!,fermion_shiftB!,add!,set_wing_fermi!,WdagWx!,apply_periodicity,Ddagx!

    # struct ParWilsonFermion
    #     wilson::WilsonFermion
    #     # extra fields that we need here
    # end
    # @forward (ParWilsonFermion, :wilson) WilsonFermion

    ############################################################################
    ################################# Structs ##################################
    ############################################################################


    """
    Parallel Wilson Fermion implementation. Very similar to WilsonFermion.jl, but implements multithreading
    for the loops.

    TODO think about if there's a better way to store the fermions which is compatible with Julia's
        column major format
    """
    struct ParWilsonFermion <: FermionFields
        NC::Int64
        NX::Int64
        NY::Int64
        NZ::Int64
        NT::Int64
        ND::Int64                                       # 2 or 4
        f::Array{ComplexF64,6}

        γ::Array{ComplexF64,3}
        rplusγ::Array{ComplexF64,3}
        rminusγ::Array{ComplexF64,3}
        hop::Float64                                    # Hopping parameter
        r::Float64                                      # Wilson term
        hopp::Array{ComplexF64,1}                       # TODO what are hopp and hopm? hopping in the + and - directions?
        hopm::Array{ComplexF64,1}
        eps::Float64
        Dirac_operator::String
        MaxCGstep::Int64
        BoundaryCondition::Array{Int8,1}
    end

    ############################################################################
    ############################# Basic utilities ##############################
    ############################################################################

    """
    Total length of a ParWilsonFermion = Nc * vol * Nd.
    """
    function Base.length(x::ParWilsonFermion)
        return x.NC*x.NX*x.NY*x.NZ*x.NT*x.ND
    end

    """
    Size of a ParWilsonFermion = (Nc, Nx, Ny, Nz, Nt, Nd).
    """
    function Base.size(x::ParWilsonFermion)
        return (x.NC, x.NX, x.NY, x.NZ, x.NT, x.ND)
    end

    """
    Iterator over ParWilsonFermion.
    """
    function Base.iterate(x::ParWilsonFermion, i::Int = 1)
        i == length(x.f)+1 && return nothing
        return (x.f[i], i + 1)
    end

    """
    Constructor for ParWilsonFermion.
    """
    function ParWilsonFermion(w::WilsonFermion)
        w_cp = deepcopy(w)
        return ParWilsonFermion(w_cp.NC, w_cp.NX, w_cp.NY, w_cp.NZ, w_cp.NT, 4, w_cp.f, w_cp.γ, w_cp.rplusγ, w_cp.rminusγ,
            w_cp.hop, w_cp.r, w_cp.hopp, w_cp.hopm, w_cp.eps, w_cp.Dirac_operator, w_cp.MaxCGstep, w_cp.BoundaryCondition)
    end

    """
    Constructor for ParWilsonFermion.
    """
    function ParWilsonFermion(NC, NX, NY, NZ, NT, ND, fparam::FermiActionParam, BoundaryCondition)
        r = fparam.r
        hop = fparam.hop
        eps = fparam.eps
        MaxCGstep = fparam.MaxCGstep
        return ParWilsonFermion(NC, NX, NY, NZ, NT, ND, r, hop, eps, MaxCGstep, BoundaryCondition)
    end

    """
    Constructor for ParWilsonFermion.
    """
    function ParWilsonFermion(NC, NX, NY, NZ, NT, ND, r, hop, eps, MaxCGstep, BoundaryCondition)
        γ,rplusγ,rminusγ = mk_gamma(r)
        hopp = zeros(ComplexF64,4)
        hopm = zeros(ComplexF64,4)
        hopp .= hop
        hopm .= hop
        Dirac_operator = "Wilson"
        return ParWilsonFermion(NC, NX, NY, NZ, NT, ND, zeros(ComplexF64, NC, NX+2, NY+2, NZ+2, NT+2, ND),
            γ, rplusγ, rminusγ, hop, r, hopp, hopm, eps, Dirac_operator, MaxCGstep, BoundaryCondition)
    end

    """
    Constructor for ParWilsonFermion.
    """
    function ParWilsonFermion(NC, NX, NY, NZ, NT, ND, γ, rplusγ, rminusγ, hop, r, hopp, hopm, eps, fermion, MaxCGstep, BoundaryCondition)
        return ParWilsonFermion(NC, NX, NY, NZ, NT, ND, zeros(ComplexF64,NC,NX+2,NY+2,NZ+2,NT+2,4),
                    γ, rplusγ, rminusγ, hop, r, hopp, hopm, eps, fermion,MaxCGstep,BoundaryCondition)
    end

    """
    Gets a halfspinor with the same parameters as an arbitrary ParWilsonFermion.
    """
    function get_halfspinor(x::ParWilsonFermion)
        return ParWilsonFermion(x.NC, x.NX, x.NY, x.NZ, x.NT, 2, x.r, x.hop, x.eps, x.MaxCGstep, x.BoundaryCondition)
    end

    """
    Sets the (i1, i2, i3, i4, i5, i6) component of x.f (the fermion field) to v. Note that x.f is 0-indexed in the
    middle and 1-indexed in the color and Dirac components.
    """
    function Base.setindex!(x::ParWilsonFermion, v, i1, i2, i3, i4, i5, i6)
        x.f[i1, i2 + 1, i3 + 1, i4 + 1, i5 + 1, i6] = v
    end

    """
    Sets the ii'th component (ii is a vectorized index) of x.f to v.
    """
    function Base.setindex!(x::ParWilsonFermion, v, ii)
        ic = (ii - 1) % x.NC + 1
        iii = (ii - ic) ÷ x.NC
        ix = iii % x.NX + 1
        iii = (iii - ix + 1) ÷ x.NX
        iy = iii % x.NY + 1
        iii = (iii - iy + 1) ÷ x.NY
        iz = iii % x.NZ + 1
        iii = (iii - iz + 1) ÷ x.NZ
        it = iii % x.NT + 1
        iii = (iii - it + 1) ÷ x.NT
        ialpha = iii + 1
        x[ic, ix, iy, iz, it, ialpha] = v
    end

    """
    Gets the (i1, i2, i3, i4, i5, i6) component of x.f (the fermion field).
    """
    function Base.getindex(x::ParWilsonFermion, i1, i2, i3, i4, i5, i6)
        return x.f[i1, i2 .+ 1, i3 .+ 1, i4 .+ 1, i5 .+ 1, i6]
    end

    """
    Gets the ii'th component (ii is a vectorized index) of x.f.
    """
    function Base.getindex(x::ParWilsonFermion, ii)
        ic = (ii - 1) % x.NC + 1
        iii = (ii - ic) ÷ x.NC
        ix = iii % x.NX + 1
        iii = (iii - ix + 1) ÷ x.NX
        iy = iii % x.NY + 1
        iii = (iii - iy + 1) ÷ x.NY
        iz = iii % x.NZ + 1
        iii = (iii - iz + 1) ÷ x.NZ
        it = iii % x.NT + 1
        iii = (iii - it + 1) ÷ x.NT
        ialpha = iii + 1
        return x[ic, ix, iy, iz, it, ialpha]
    end

    ############################################################################
    ########################### Algebraic operations ###########################
    ############################################################################

    #=
    We likely won't need to re-implement all of these. We should think about ways to multithread
    these functions: we can probably just do it at the level of the for-loops (embarassingly
    parallel problem) since a lot of the data here seems independent.

    For the previous set of functions, I reimplemented each function for both ParWilsonFermion and
    for ParWilsonAction, but for these it might be more of a pain than it's worth (there are a lot of
    functions in the WilsonFermion.jl file!)
    =#

    function Base.:*(a::ParWilsonFermion, b::ParWilsonFermion)
        c = 0.0im
        for α=1:4
            for it=1:a.NT
                for iz=1:a.NZ
                    for iy=1:a.NY
                        for ix=1:a.NX
                            @simd for ic=1:a.NC
                                c+= conj(a[ic,ix,iy,iz,it,α])*b[ic,ix,iy,iz,it,α]
                            end
                        end
                    end
                end
            end
        end
        return c
    end

    function Base.:+(a::ParWilsonFermion, b::ParWilsonFermion)
        c = similar(a)
        for α=1:4
            for it=1:a.NT
                for iz=1:a.NZ
                    for iy=1:a.NY
                        for ix=1:a.NX
                            @simd for ic=1:a.NC
                                c[ic,ix,iy,iz,it,α]=a[ic,ix,iy,iz,it,α]+b[ic,ix,iy,iz,it,α]
                            end
                        end
                    end
                end
            end
        end
        return c
    end

    function Base.fill!(a::ParWilsonFermion, x::T) where T <: Number
        for α=1:4
            for it=1:a.NT
                for iz=1:a.NZ
                    for iy=1:a.NY
                        for ix=1:a.NX
                            @simd for ic=1:a.NC
                                a[ic,ix,iy,iz,it,α]=x
                            end
                        end
                    end
                end
            end
        end
    end

    function Base.:-(a::ParWilsonFermion, b::ParWilsonFermion)
        return a+(-1)*b
    end

    function Base.:*(a::T,b::ParWilsonFermion) where T <: Number
        c = similar(b)
        for α=1:4
            for it=1:b.NT
                for iz=1:b.NZ
                    for iy=1:b.NY
                        for ix=1:b.NX
                            @simd for ic=1:b.NC
                                c[ic,ix,iy,iz,it,α] = a*b[ic,ix,iy,iz,it,α]
                            end
                        end
                    end
                end
            end
        end
        return c
    end

    function LinearAlgebra.axpy!(a::T,X::ParWilsonFermion,Y::ParWilsonFermion) where T <: Number #Y = a*X+Y
        for α=1:4
            for it=1:X.NT
                for iz=1:X.NZ
                    for iy=1:X.NY
                        for ix=1:X.NX
                            @simd for ic=1:X.NC
                                Y[ic,ix,iy,iz,it,α] += a*X[ic,ix,iy,iz,it,α]
                            end
                        end
                    end
                end
            end
        end
        return X
    end

    """
    Scalar multiplication of a ParWilsonFermion.
    """
    function LinearAlgebra.rmul!(a::ParWilsonFermion,b::T) where T <: Number
        @threads for α=1:a.ND
            for it=1:a.NT
                for iz=1:a.NZ
                    for iy=1:a.NY
                        for ix=1:a.NX
                            @simd for ic=1:a.NC
                                a[ic,ix,iy,iz,it,α] = b*a[ic,ix,iy,iz,it,α]
                            end
                        end
                    end
                end
            end
        end
        return a
    end

    """
    Dummy rmul2! operator to use for testing the multithreading. TODO change the way ParWilsonFermions store their
    data and see what happens.
    """
    function rmul2!(a::ParWilsonFermion,b::T) where T <: Number
        @threads for it=1:a.NT
            for α=1:a.ND
                for iz=1:a.NZ
                    for iy=1:a.NY
                        for ix=1:a.NX
                            @simd for ic=1:a.NC
                                a[ic,ix,iy,iz,it,α] = b*a[ic,ix,iy,iz,it,α]
                            end
                        end
                    end
                end
            end
        end
        return a
    end

    function Base.similar(x::ParWilsonFermion)
        return ParWilsonFermion(x.NC,x.NX,x.NY,x.NZ,x.NT,
                    4,x.r,x.hop,x.eps,x.MaxCGstep,x.BoundaryCondition)
    end

    # TODO these are likely broken and we should reimplement
    function LinearAlgebra.mul!(xout::ParWilsonFermion,A::AbstractMatrix,x::ParWilsonFermion)
        NX = x.NX
        NY = x.NY
        NZ = x.NZ
        NT = x.NT
        NC = x.NC
        for ic=1:NC
            for it=1:NT
                for iz=1:NZ
                    for iy=1:NY
                        @simd for ix=1:NX
                                e1 = x[ic,ix,iy,iz,it,1]
                                e2 = x[ic,ix,iy,iz,it,2]
                                e3 = x[ic,ix,iy,iz,it,3]
                                e4 = x[ic,ix,iy,iz,it,4]

                                xout[ic,ix,iy,iz,it,1] = A[1,1]*e1+A[1,2]*e2+A[1,3]*e3+A[1,4]*e4
                                xout[ic,ix,iy,iz,it,2] = A[2,1]*e1+A[2,2]*e2+A[2,3]*e3+A[2,4]*e4
                                xout[ic,ix,iy,iz,it,3] = A[3,1]*e1+A[3,2]*e2+A[3,3]*e3+A[3,4]*e4
                                xout[ic,ix,iy,iz,it,4] = A[4,1]*e1+A[4,2]*e2+A[4,3]*e3+A[4,4]*e4
                        end
                    end
                end
            end
        end
    end

    function LinearAlgebra.mul!(xout::ParWilsonFermion,x::ParWilsonFermion,A::AbstractMatrix)
        NX = x.NX
        NY = x.NY
        NZ = x.NZ
        NT = x.NT
        NC = x.NC

        for ic=1:NC
            for it=1:NT
                for iz=1:NZ
                    for iy=1:NY
                        @simd for ix=1:NX
                            e1 = x[ic,ix,iy,iz,it,1]
                            e2 = x[ic,ix,iy,iz,it,2]
                            e3 = x[ic,ix,iy,iz,it,3]
                            e4 = x[ic,ix,iy,iz,it,4]

                            xout[ic,ix,iy,iz,it,1] = A[1,1]*e1+A[2,1]*e2+A[3,1]*e3+A[4,1]*e4
                            xout[ic,ix,iy,iz,it,2] = A[1,2]*e1+A[2,2]*e2+A[3,2]*e3+A[4,2]*e4
                            xout[ic,ix,iy,iz,it,3] = A[1,3]*e1+A[2,3]*e2+A[3,3]*e3+A[4,3]*e4
                            xout[ic,ix,iy,iz,it,4] = A[1,4]*e1+A[2,4]*e2+A[3,4]*e3+A[4,4]*e4
                        end
                    end
                end
            end
        end
    end

    """
    Multiplies the Dirac structure of a matrix with that of a FermionField. This should
    replace LatticeQCD.FermionField.mul!, which does not work.
    """
    function mul_dirac!(ferm::FermionFields, A::Matrix{ComplexF64}, x::FermionFields)
        NX = x.NX
        NY = x.NY
        NZ = x.NZ
        NT = x.NT
        NC = x.NC
        for ic=1:NC
            for it=1:NT + 2
                for iz=1:NZ + 2
                    for iy=1:NY + 2
                        for ix=1:NX + 2
                            vec = x.f[ic, ix, iy, iz, it, :]
                            tmp = A * vec
                            ferm.f[ic, ix, iy, iz, it, :] = tmp
                        end
                    end
                end
            end
        end
        return
    end

    """
    Adds two FermionFields together at a given spacetime point coords.
    """
    function add_pt!(c::FermionFields, alpha::Number, a::FermionFields, beta::Number, b::FermionFields, coords::SVector{4, Int64})
        Nc, n2, n3, n4, n5, Nd = size(a.f)
        ix = coords[1] + 1
        iy = coords[2] + 1
        iz = coords[3] + 1
        it = coords[4] + 1

        for ialpha = 1:Nd
            @simd for ic = 1:Nc
                c.f[ic, ix, iy, iz, it, ialpha] += alpha * a.f[ic, ix, iy, iz, it, ialpha] + beta * b.f[ic, ix, iy, iz, it, ialpha]
            end
        end
        return
    end

    """
    Adds two FermionFields together in a range of spacetime point coords.
    """
    function add_rg!(c::FermionFields, alpha::Number, a::FermionFields, beta::Number, b::FermionFields, coords_min::SVector{4, Int64}, coords_max::SVector{4, Int64})
        ix_min = coords_min[1] + 1
        iy_min = coords_min[2] + 1
        iz_min = coords_min[3] + 1
        it_min = coords_min[4] + 1
        ix_max = coords_max[1] + 1
        iy_max = coords_max[2] + 1
        iz_max = coords_max[3] + 1
        it_max = coords_max[4] + 1

        c.f[:, ix_min:ix_max, iy_min:iy_max, iz_min:iz_max, it_min:it_max, :] .+= alpha .* a.f[:, ix_min:ix_max, iy_min:iy_max, iz_min:iz_max, it_min:it_max, :] .+ beta .* b.f[:, ix_min:ix_max, iy_min:iy_max, iz_min:iz_max, it_min:it_max, :]
        return
    end

    function substitute_fermion!(H, j, x::ParWilsonFermion)
        i = 0
        for ialpha = 1:4
            for it=1:x.NT
                for iz=1:x.NZ
                    for iy=1:x.NY
                        for ix=1:x.NX
                            @simd for ic=1:x.NC
                                i += 1
                                H[i,j] = x[ic,ix,iy,iz,it,ialpha]
                            end
                        end
                    end
                end
            end
        end
    end

    ############################################################################
    ########################### Shifting operations ############################
    ############################################################################

    #=
    These are gauge invariant shifting operations. They'll likely be useful to reimplement: notice that the type
    T here is a gauge field, so u::Array{T, 1} is just a gauge field u[μ], where μ ∈ {1, 2, 3, 4}: the multiplication
    by u[μ] serves to make the shifted gauge field invariant. We'll likely need to implement all these functions for
    ParWilsonFermions.
    =#

    function fermion_shift!(b::ParWilsonFermion,u::Array{T,1},μ::Int,a::ParWilsonFermion) where T <: SU3GaugeFields
        if μ == 0
            substitute!(b,a)
            return
        end

        NX = a.NX
        NY = a.NY
        NZ = a.NZ
        NT = a.NT
        ND = a.ND
        NC = 3

        if μ > 0
            for ialpha=1:ND
                for it=1:NT
                    it1 = it + ifelse(μ ==4,1,0)
                    for iz=1:NZ
                        iz1 = iz + ifelse(μ ==3,1,0)
                        for iy=1:NY
                            iy1 = iy + ifelse(μ ==2,1,0)
                            @simd for ix=1:NX
                                ix1 = ix + ifelse(μ ==1,1,0)

                                b[1,ix,iy,iz,it,ialpha] = u[μ][1,1,ix,iy,iz,it]*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                            u[μ][1,2,ix,iy,iz,it]*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                            u[μ][1,3,ix,iy,iz,it]*a[3,ix1,iy1,iz1,it1,ialpha]

                                b[2,ix,iy,iz,it,ialpha] = u[μ][2,1,ix,iy,iz,it]*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                            u[μ][2,2,ix,iy,iz,it]*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                            u[μ][2,3,ix,iy,iz,it]*a[3,ix1,iy1,iz1,it1,ialpha]

                                b[3,ix,iy,iz,it,ialpha] = u[μ][3,1,ix,iy,iz,it]*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                            u[μ][3,2,ix,iy,iz,it]*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                            u[μ][3,3,ix,iy,iz,it]*a[3,ix1,iy1,iz1,it1,ialpha]
                            end
                        end
                    end
                end
            end

        elseif μ < 0
            for ialpha =1:ND
                for it=1:NT
                    it1 = it - ifelse(-μ ==4,1,0)
                    for iz=1:NZ
                        iz1 = iz - ifelse(-μ ==3,1,0)
                        for iy=1:NY
                            iy1 = iy - ifelse(-μ ==2,1,0)
                            @simd for ix=1:NX
                                ix1 = ix - ifelse(-μ ==1,1,0)

                                b[1,ix,iy,iz,it,ialpha] = conj(u[-μ][1,1,ix1,iy1,iz1,it1])*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                            conj(u[-μ][2,1,ix1,iy1,iz1,it1])*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                            conj(u[-μ][3,1,ix1,iy1,iz1,it1])*a[3,ix1,iy1,iz1,it1,ialpha]

                                b[2,ix,iy,iz,it,ialpha] = conj(u[-μ][1,2,ix1,iy1,iz1,it1])*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                            conj(u[-μ][2,2,ix1,iy1,iz1,it1])*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                            conj(u[-μ][3,2,ix1,iy1,iz1,it1])*a[3,ix1,iy1,iz1,it1,ialpha]

                                b[3,ix,iy,iz,it,ialpha] = conj(u[-μ][1,3,ix1,iy1,iz1,it1])*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                            conj(u[-μ][2,3,ix1,iy1,iz1,it1])*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                            conj(u[-μ][3,3,ix1,iy1,iz1,it1])*a[3,ix1,iy1,iz1,it1,ialpha]
                            end
                        end
                    end
                end
            end
        end

    end

    """
    Only shifts the fermion field at a specific point
    """
    function fermion_shift_pt!(b::ParWilsonFermion, u::Array{T,1}, coords::SVector{4, Int64}, μ::Int, a::ParWilsonFermion) where T <: SU3GaugeFields
        if μ == 0
            substitute!(b,a)
            return
        end
        ND = a.ND
        NC = 3

        ix = coords[1] + 1
        iy = coords[2] + 1
        iz = coords[3] + 1
        it = coords[4] + 1

        if μ > 0
            for ialpha=1:ND
                it1 = it + ifelse(μ == 4,1,0)
                iz1 = iz + ifelse(μ == 3,1,0)
                iy1 = iy + ifelse(μ == 2,1,0)
                ix1 = ix + ifelse(μ == 1,1,0)

                # NOTE if NDW != 1, will have to change padding on u[μ] to ix ± NDW
                b.f[1,ix,iy,iz,it,ialpha] = u[μ].g[1,1,ix,iy,iz,it] * a.f[1,ix1,iy1,iz1,it1,ialpha] +
                                            u[μ].g[1,2,ix,iy,iz,it] * a.f[2,ix1,iy1,iz1,it1,ialpha] +
                                            u[μ].g[1,3,ix,iy,iz,it] * a.f[3,ix1,iy1,iz1,it1,ialpha]
                b.f[2,ix,iy,iz,it,ialpha] = u[μ].g[2,1,ix,iy,iz,it] * a.f[1,ix1,iy1,iz1,it1,ialpha] +
                                            u[μ].g[2,2,ix,iy,iz,it] * a.f[2,ix1,iy1,iz1,it1,ialpha] +
                                            u[μ].g[2,3,ix,iy,iz,it] * a.f[3,ix1,iy1,iz1,it1,ialpha]
                b.f[3,ix,iy,iz,it,ialpha] = u[μ].g[3,1,ix,iy,iz,it] * a.f[1,ix1,iy1,iz1,it1,ialpha] +
                                            u[μ].g[3,2,ix,iy,iz,it] * a.f[2,ix1,iy1,iz1,it1,ialpha] +
                                            u[μ].g[3,3,ix,iy,iz,it] * a.f[3,ix1,iy1,iz1,it1,ialpha]
            end

        elseif μ < 0
            for ialpha =1:ND
                it1 = it - ifelse(-μ ==4,1,0)
                iz1 = iz - ifelse(-μ ==3,1,0)
                iy1 = iy - ifelse(-μ ==2,1,0)
                ix1 = ix - ifelse(-μ ==1,1,0)

                b.f[1,ix,iy,iz,it,ialpha] = conj(u[-μ].g[1,1,ix1,iy1,iz1,it1]) * a.f[1,ix1,iy1,iz1,it1,ialpha] +
                                            conj(u[-μ].g[2,1,ix1,iy1,iz1,it1]) * a.f[2,ix1,iy1,iz1,it1,ialpha] +
                                            conj(u[-μ].g[3,1,ix1,iy1,iz1,it1]) * a.f[3,ix1,iy1,iz1,it1,ialpha]
                b.f[2,ix,iy,iz,it,ialpha] = conj(u[-μ].g[1,2,ix1,iy1,iz1,it1]) * a.f[1,ix1,iy1,iz1,it1,ialpha] +
                                            conj(u[-μ].g[2,2,ix1,iy1,iz1,it1]) * a.f[2,ix1,iy1,iz1,it1,ialpha] +
                                            conj(u[-μ].g[3,2,ix1,iy1,iz1,it1]) * a.f[3,ix1,iy1,iz1,it1,ialpha]
                b.f[3,ix,iy,iz,it,ialpha] = conj(u[-μ].g[1,3,ix1,iy1,iz1,it1]) * a.f[1,ix1,iy1,iz1,it1,ialpha] +
                                            conj(u[-μ].g[2,3,ix1,iy1,iz1,it1]) * a.f[2,ix1,iy1,iz1,it1,ialpha] +
                                            conj(u[-μ].g[3,3,ix1,iy1,iz1,it1]) * a.f[3,ix1,iy1,iz1,it1,ialpha]
            end
        end
        return
    end

    function fermion_shift_gamma!(b::ParWilsonFermion,u::Array{T,1},μ::Int,a::ParWilsonFermion) where T <: SU3GaugeFields
        if μ == 0
            substitute!(b,a)
            return
        end

        NX = a.NX
        NY = a.NY
        NZ = a.NZ
        NT = a.NT
        NC = 3

        if μ > 0
            n6 = size(a.f)[6]
            for ialpha=1:4
                for it=1:NT
                    it1 = it + ifelse(μ ==4,1,0)
                    for iz=1:NZ
                        iz1 = iz + ifelse(μ ==3,1,0)
                        for iy=1:NY
                            iy1 = iy + ifelse(μ ==2,1,0)
                            @simd for ix=1:NX
                                ix1 = ix + ifelse(μ ==1,1,0)

                                b[1,ix,iy,iz,it,ialpha] = u[μ][1,1,ix,iy,iz,it]*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                            u[μ][1,2,ix,iy,iz,it]*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                            u[μ][1,3,ix,iy,iz,it]*a[3,ix1,iy1,iz1,it1,ialpha]

                                b[2,ix,iy,iz,it,ialpha] = u[μ][2,1,ix,iy,iz,it]*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                            u[μ][2,2,ix,iy,iz,it]*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                            u[μ][2,3,ix,iy,iz,it]*a[3,ix1,iy1,iz1,it1,ialpha]

                                b[3,ix,iy,iz,it,ialpha] = u[μ][3,1,ix,iy,iz,it]*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                            u[μ][3,2,ix,iy,iz,it]*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                            u[μ][3,3,ix,iy,iz,it]*a[3,ix1,iy1,iz1,it1,ialpha]
                            end
                        end
                    end
                end
            end

        elseif μ < 0
            for ialpha =1:4
                for it=1:NT
                    it1 = it - ifelse(-μ ==4,1,0) #idel[4]
                    for iz=1:NZ
                        iz1 = iz - ifelse(-μ ==3,1,0) #idel[3]
                        for iy=1:NY
                            iy1 = iy - ifelse(-μ ==2,1,0)  #idel[2]
                            @simd for ix=1:NX
                                ix1 = ix - ifelse(-μ ==1,1,0) #idel[1]

                                b[1,ix,iy,iz,it,ialpha] = conj(u[-μ][1,1,ix1,iy1,iz1,it1])*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                            conj(u[-μ][2,1,ix1,iy1,iz1,it1])*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                            conj(u[-μ][3,1,ix1,iy1,iz1,it1])*a[3,ix1,iy1,iz1,it1,ialpha]

                                b[2,ix,iy,iz,it,ialpha] = conj(u[-μ][1,2,ix1,iy1,iz1,it1])*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                            conj(u[-μ][2,2,ix1,iy1,iz1,it1])*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                            conj(u[-μ][3,2,ix1,iy1,iz1,it1])*a[3,ix1,iy1,iz1,it1,ialpha]

                                b[3,ix,iy,iz,it,ialpha] = conj(u[-μ][1,3,ix1,iy1,iz1,it1])*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                            conj(u[-μ][2,3,ix1,iy1,iz1,it1])*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                            conj(u[-μ][3,3,ix1,iy1,iz1,it1])*a[3,ix1,iy1,iz1,it1,ialpha]
                            end
                        end
                    end
                end
            end
        end
    end

    ############################################################################
    ######################### FermionFields operations #########################
    ############################################################################

    """
    set_wing_fermi! function (sets the padding to the appropriate boundary condition for the field) which actually works.
    """
    function set_wing_fermi_correct!(a::FermionFields)
        NT = a.NT
        NZ = a.NZ
        NY = a.NY
        NX = a.NX
        NC = a.NC

        #!  X-direction
        for ialpha=1:4
            for it=1:NT
                for iz = 1:NZ
                    for iy=1:NY
                        for k=1:NC
                            a.f[k, 1, iy + 1, iz + 1, it + 1, ialpha] = a.BoundaryCondition[1] * a.f[k, NX + 1, iy + 1, iz + 1, it + 1, ialpha]
                            a.f[k, NX + 2, iy + 1, iz + 1, it + 1, ialpha] = a.BoundaryCondition[1] * a.f[k, 2, iy + 1, iz + 1, it + 1, ialpha]
                        end
                    end
                end
            end
        end

        #Y-direction
        for ialpha = 1:4
            for it=1:NT
                for iz=1:NZ
                    for ix=1:NX
                        for k=1:NC
                            a.f[k, ix + 1, 1, iz + 1,it + 1, ialpha] = a.BoundaryCondition[2] * a.f[k, ix + 1, NY + 1, iz + 1, it + 1, ialpha]
                            a.f[k, ix + 1, NY + 2, iz + 1, it + 1, ialpha] = a.BoundaryCondition[2] * a.f[k, ix + 1, 2, iz + 1, it + 1, ialpha]
                        end
                    end
                end
            end
        end

        for ialpha=1:4
            # Z-direction
            for it=1:NT
                for iy=1:NY
                    for ix=1:NX
                        for k=1:NC
                            a.f[k, ix + 1, iy + 1, 1, it + 1,ialpha] = a.BoundaryCondition[3] * a.f[k, ix + 1, iy + 1, NZ + 1, it + 1, ialpha]
                            a.f[k, ix + 1, iy + 1, NZ + 2, it + 1, ialpha] = a.BoundaryCondition[3] * a.f[k, ix + 1, iy + 1, 2, it + 1, ialpha]
                        end
                    end
                end
            end

            #T-direction
            for iz=1:NZ
                for iy=1:NY
                    for ix=1:NX
                        for k=1:NC
                            a.f[k, ix + 1, iy + 1, iz + 1, 1, ialpha] = a.BoundaryCondition[4] * a.f[k, ix + 1, iy + 1, iz + 1, NT + 1, ialpha]
                            a.f[k, ix + 1, iy + 1, iz + 1, NT + 2, ialpha] = a.BoundaryCondition[4] * a.f[k, ix + 1, iy + 1, iz + 1, 2, ialpha]
                        end
                    end
                end
            end
        end
    end

    """
    Pads the front and end of each lattice direction with the appropriate boundary condition.
    """
    function set_wing_fermi_threads!(a::ParWilsonFermion)
        NT = a.NT
        NZ = a.NZ
        NY = a.NY
        NX = a.NX
        NC = a.NC
        ND = a.ND

        @threads for ialpha = 1:ND
            for it = 1:NT

                for iz = 1:NZ       #!  X-direction
                    for iy = 1:NY
                        for k = 1:NC
                            a[k,0,iy,iz,it,ialpha] = a.BoundaryCondition[1]*a[k,NX,iy,iz,it,ialpha]
                            a[k,NX+1,iy,iz,it,ialpha] = a.BoundaryCondition[1]*a[k,1,iy,iz,it,ialpha]
                        end
                    end
                end

                for iz = 1:NZ       #! Y-direction
                    for ix = 1:NX
                        for k = 1:NC
                            a[k,ix,0,iz,it,ialpha] = a.BoundaryCondition[2]*a[k,ix,NY,iz,it,ialpha]
                            a[k,ix,NY+1,iz,it,ialpha] = a.BoundaryCondition[2]*a[k,ix,1,iz,it,ialpha]
                        end
                    end
                end

                for it = 1:NT       # Z-direction
                    for iy = 1:NY
                        for ix = 1:NX
                            for k = 1:NC
                                a[k,ix,iy,0,it,ialpha] = a.BoundaryCondition[3]*a[k,ix,iy,NZ,it,ialpha]
                                a[k,ix,iy,NZ+1,it,ialpha] = a.BoundaryCondition[3]*a[k,ix,iy,1,it,ialpha]
                            end
                        end
                    end
                end

            end

            for iz = 1:NZ           #T-direction
                for iy = 1:NY
                    for ix = 1:NX
                        for k = 1:NC
                            a[k,ix,iy,iz,0,ialpha] = a.BoundaryCondition[4]*a[k,ix,iy,iz,NT,ialpha]
                            a[k,ix,iy,iz,NT+1,ialpha] = a.BoundaryCondition[4]*a[k,ix,iy,iz,1,ialpha]
                        end
                    end
                end
            end

        end

    end


    ############################################################################
    ############################# Wilson operators #############################
    ############################################################################

    function Dx_serial!(xout::ParWilsonFermion, U::Array{G,1}, x::ParWilsonFermion, temps::Array{T,1}) where  {T <: FermionFields, G <: GaugeFields}
        temp1 = temps[1]
        temp2 = temps[2]
        temp3 = temps[3]
        temp4 = temps[4]
        temp = temps[5]

        clear!(temp)
        set_wing_fermi_correct!(x)
        for ν=1:4
            # shift + project
            fermion_shift!(temp1, U, ν, x)
            mul_dirac!(temp3, x.rminusγ[:, :, ν], temp1)

            fermion_shift!(temp2, U, -ν, x)
            mul_dirac!(temp4, x.rplusγ[:, :, ν], temp2)

            add!(temp, 0.5, temp3, 0.5, temp4)
        end

        clear!(xout)
        add!(xout, 1/(2*x.hop), x, -1, temp)            # Off by a factor of 2κ?
        return
    end

    """
    Evaluates Dx! using the half-spinor projection.
    """
    function Dx_halfspinor!(xout::ParWilsonFermion, U::Array{G, 1}, x::ParWilsonFermion, full_temps::Array{T, 1},
                    temps::Array{T, 1}) where  {T <: FermionFields, G <: GaugeFields}
        temp1 = temps[1]            # temps must be a matrix of Nd = 2 halfspinors.
        temp2 = temps[2]
        temp3 = temps[3]
        temp4 = temps[4]

        full_temp1 = full_temps[1]
        full_temp2 = full_temps[2]
        temp = full_temps[3]

        clear!(temp)
        set_wing_fermi_correct!(x)

        for ν=1:4                   # Workflow: project, shift, recon
            proj_halfspinor!(temp1, x, ν, false)
            fermion_shift!(temp3, U, ν, temp1)
            recon_halfspinor!(full_temp1, temp3, ν, false)

            proj_halfspinor!(temp2, x, ν, true)
            fermion_shift!(temp4, U, -ν, temp2)
            recon_halfspinor!(full_temp2, temp4, ν, true)

            add!(temp, 0.5, full_temp1, 0.5, full_temp2)
        end

        clear!(xout)
        add!(xout, 1/(2*x.hop), x, -1, temp)
        return
    end

    """
    Evaluates Dx! using the half-spinor projection and multithreading. Multithreads pointwise on each
    spacetime point on the lattice.
    """
    function Dx_halfspinor_threadall!(xout::ParWilsonFermion, U::Array{G, 1}, x::ParWilsonFermion, full_temps::Array{T, 1},
                    temps::Array{T, 1}) where  {T <: FermionFields, G <: GaugeFields}
        temp1 = temps[1]            # temps must be a matrix of Nd = 2 halfspinors.
        temp2 = temps[2]
        temp3 = temps[3]
        temp4 = temps[4]

        full_temp1 = full_temps[1]
        full_temp2 = full_temps[2]
        temp = full_temps[3]

        clear!(temp)
        set_wing_fermi_correct!(x)
        Nt = x.NT; Nz = x.NZ; Ny = x.NY; Nx = x.NX
        @threads for it = 1:Nt
            for iz = 1:Nz
                for iy = 1:Ny
                    for ix = 1:Nx
                        coords = @SVector [ix, iy, iz, it]
                        for ν=1:4                   # Workflow: project, shift, recon
                            proj_halfspinor_pt!(temp1, x, coords, ν, false)
                            fermion_shift_pt!(temp3, U, coords, ν, temp1)
                            recon_halfspinor_pt!(full_temp1, temp3, coords, ν, false)

                            proj_halfspinor_pt!(temp2, x, coords, ν, true)
                            fermion_shift_pt!(temp4, U, coords, -ν, temp2)
                            recon_halfspinor_pt!(full_temp2, temp4, coords, ν, true)

                            add_pt!(temp, 0.5, full_temp1, 0.5, full_temp2, coords)
                        end
                    end
                end
            end
        end

        clear!(xout)
        add!(xout, 1/(2*x.hop), x, -1, temp)
        return
    end

    """
    Splits x into prod geom different container fermion fields. Each container in xtemps should be of size
    length(x) / nsplit.

    Parameters
    ----------
    xtemps::Array{T, 1}
        Temporary x to split into. Size of x should be prod geom.
    x::ParWilsonFermion
        Fermion to split.
    geom::Array{Integer, 1}
        4D array with relative geometry to split.
    """
    function split_x_geom!(xtemps::Array{ParWilsonFermion, 1}, x::ParWilsonFermion, geom::Array{Integer, 1})
        gx = geom[1]; gy = geom[2]; gz = geom[3]; gt = geom[4]
        Nx = x.NX; Ny = x.NY; Nz = x.NZ; Nt = x.NT; Nd = x.ND; Nc = x.NC
        Nblockx = Nx / gx; Nblocky = Ny / gy; Nblockz = Nz / gz; Nblockt = Nt / gt;
        n_blocks = gx * gy * gz * gt

        blk_idx = 1
        for blkx = 1 : gx
            for blky = 1 : gy
                for blkz = 1 : gz
                    for blkt = 1 : gt                   # iterate over blocks

                        tmp = xtemps[blk_idx]
                        for iα = 1 : Nd                 # populate tmp
                            for xx = 1 : Nblockx
                                iix = xx + Nblockx * (blkx - 1) + 1            # +1 is for padding
                                for yy = 1 : Nblocky
                                    iiy = yy + Nblocky * (blky - 1) + 1
                                    for zz = 1 : Nblockz
                                        iiz = zz + Nblockz * (blkz - 1) + 1
                                        for tt = 1 : Nblockt
                                            iit = tt + Nblockt * (blkt - 1) + 1
                                            for k = 1 : Nc
                                                 tmp.f[k, xx, yy, zz, tt, iα] = x.f[k, iix, iiy, iiz, iit, iα]
                                                 # TODO have to give each fermion the correct bcs
                                            end
                                        end
                                    end
                                end
                            end
                        end
                        blk_idx += 1
                    end
                end
            end
        end
        return
    end

    """
    Reconstructs a Fermion field from pieces split up according to geom.
    """
    function gather_x_geom!(xout::ParWilsonFermion, pieces::Array{ParWilsonFermion, 1}, geom::Array{Integer, 1})
        # TODO function stub that we did not have time to implement
    end

    """
    Distributed implementation of Dx!. Lexigraphically breaks the lattice into n_threads sublattices and
    calls Dx_halfspinor! on each one.
    """
    function Dx_block!(xout::ParWilsonFermion, U::Array{G, 1}, x::ParWilsonFermion, blk_out::Array{ParWilsonFermion, 1}, blks::Array{ParWilsonFermion, 1}, temps::Array{ParWilsonFermion, 1}, geom::Array{Integer, 1}) where  G <: GaugeFields

        n_threads = Threads.nthreads()
        # actually get n_ranks from MPI
        split_x_geom!(blks, x, geom)                # note that blks should have the same size as prod geom
        @threads for th_idx = 1 : n_threads
            Dx_halfspinor!(blk_out[th_idx], )
        end

        clear!(temp)
        set_wing_fermi!(x)              # may want to move this out

        # TODO function stub that we did not have time to implement

        return
    end

    function Ddagx!(xout::ParWilsonFermion,U::Array{G,1},
        x::ParWilsonFermion,temps::Array{T,1}) where  {T <: FermionFields,G <: GaugeFields}
        temp = temps[4]
        temp1 = temps[1]
        temp2 = temps[2]

        clear!(temp)
        set_wing_fermi!(x)
        for ν=1:4
            fermion_shift!(temp1,U,ν,x)

            #... Dirac multiplication
            #mul!(temp1,view(x.rminusγ,:,:,ν),temp1)
            mul!(temp1,view(x.rplusγ,:,:,ν),temp1)

            #
            fermion_shift!(temp2,U,-ν,x)
            #mul!(temp2,view(x.rplusγ,:,:,ν),temp2)
            mul!(temp2,view(x.rminusγ,:,:,ν),temp2)

            add!(temp,0.5,temp1,0.5,temp2)          # TODO replace add


        end

        clear!(xout)
        add!(xout,1/(2*x.hop),x,-1,temp)
        return
    end

    ############################################################################
    ########################### TwoSpinor Operations ###########################
    ############################################################################

    """
    There are a lot of utility functions like this. We need to think about the
    best way to implement them in terms of projecting down to two-spinors and then
    reconstructing.
    """
    function mul_γ5x!(y::ParWilsonFermion, x::ParWilsonFermion)
        NX = x.NX
        NY = x.NY
        NZ = x.NZ
        NT = x.NT
        NC = x.NC
        for ic=1:NC
            for it=1:NT
                for iz=1:NZ
                    for iy=1:NY
                        @simd for ix=1:NX
                            y[ic,ix,iy,iz,it,1] = -1*x[ic,ix,iy,iz,it,1]
                            y[ic,ix,iy,iz,it,2] = -1*x[ic,ix,iy,iz,it,2]
                            y[ic,ix,iy,iz,it,3] = x[ic,ix,iy,iz,it,3]
                            y[ic,ix,iy,iz,it,4] = x[ic,ix,iy,iz,it,4]
                        end
                    end
                end
            end
        end
    end

    """
    Projects a spinor x in the μ direction onto a halfspinor.

    Parameters
    ----------
    out::ParWilsonFermion
        Output ParWilsonFermion field (ND = 2)
    x::ParWilsonFermion
        Wilson fermion field to project with ND = 4.
    μ::Integer
        Direction, should be in {1, 2, 3, 4}
    pos::Bool
        If positive or negative projector. True if positive.

    Returns
    -------
    """
    function proj_halfspinor!(out::ParWilsonFermion, x::ParWilsonFermion, μ::Integer, plus::Bool)
        if plus
            if μ == 1
                out.f[:, :, :, :, :, 1] .= x.f[:, :, :, :, :, 1] .- im .* x.f[:, :, :, :, :, 4]          # h0
                out.f[:, :, :, :, :, 2] .= x.f[:, :, :, :, :, 2] .- im .* x.f[:, :, :, :, :, 3]          # h1
            elseif μ == 2
                out.f[:, :, :, :, :, 1] .= x.f[:, :, :, :, :, 1] .- x.f[:, :, :, :, :, 4]
                out.f[:, :, :, :, :, 2] .= x.f[:, :, :, :, :, 2] .+ x.f[:, :, :, :, :, 3]
            elseif μ == 3
                out.f[:, :, :, :, :, 1] .= x.f[:, :, :, :, :, 1] .- im .* x.f[:, :, :, :, :, 3]
                out.f[:, :, :, :, :, 2] .= x.f[:, :, :, :, :, 2] .+ im .* x.f[:, :, :, :, :, 4]
            else
                out.f[:, :, :, :, :, 1] .= x.f[:, :, :, :, :, 1] .- x.f[:, :, :, :, :, 3]
                out.f[:, :, :, :, :, 2] .= x.f[:, :, :, :, :, 2] .- x.f[:, :, :, :, :, 4]
            end
        else
            if μ == 1
                out.f[:, :, :, :, :, 1] .= x.f[:, :, :, :, :, 1] .+ im .* x.f[:, :, :, :, :, 4]          # h0
                out.f[:, :, :, :, :, 2] .= x.f[:, :, :, :, :, 2] .+ im .* x.f[:, :, :, :, :, 3]          # h1
            elseif μ == 2
                out.f[:, :, :, :, :, 1] .= x.f[:, :, :, :, :, 1] .+ x.f[:, :, :, :, :, 4]
                out.f[:, :, :, :, :, 2] .= x.f[:, :, :, :, :, 2] .- x.f[:, :, :, :, :, 3]
            elseif μ == 3
                out.f[:, :, :, :, :, 1] .= x.f[:, :, :, :, :, 1] .+ im .* x.f[:, :, :, :, :, 3]
                out.f[:, :, :, :, :, 2] .= x.f[:, :, :, :, :, 2] .- im .* x.f[:, :, :, :, :, 4]
            else
                out.f[:, :, :, :, :, 1] .= x.f[:, :, :, :, :, 1] .+ x.f[:, :, :, :, :, 3]
                out.f[:, :, :, :, :, 2] .= x.f[:, :, :, :, :, 2] .+ x.f[:, :, :, :, :, 4]
            end
        end
        return
    end

    """
    Reconstructs a projected spinor y to a full spinor x in the μ direction.

    Parameters
    ----------
    out::ParWilsonFermion (ND = 4)
        Reconstructed spinor (two spinor components instead of 4.)
    y::ParWilsonFermion (ND = 2)
        TwoSpinor fermion field to reconstruct.
    μ::Integer
        Direction, should be in {1, 2, 3, 4} or {-1, -2, -3, -4}

    Returns
    -------
    ParWilsonFermion
        Reconstructed spinor (two spinor components instead of 4.)
    """
    function recon_halfspinor!(out::ParWilsonFermion, x::ParWilsonFermion, μ::Integer, plus::Bool)
        out.f[:, :, :, :, :, 1] .= x.f[:, :, :, :, :, 1]                            # h0
        out.f[:, :, :, :, :, 2] .= x.f[:, :, :, :, :, 2]                            # h1
        if plus
            if μ == 1
                out.f[:, :, :, :, :, 3] .= im .* x.f[:, :, :, :, :, 2]              # r2
                out.f[:, :, :, :, :, 4] .= im .* x.f[:, :, :, :, :, 1]              # r3
            elseif μ == 2
                out.f[:, :, :, :, :, 3] .= x.f[:, :, :, :, :, 2]
                out.f[:, :, :, :, :, 4] .= -x.f[:, :, :, :, :, 1]
            elseif μ == 3
                out.f[:, :, :, :, :, 3] .= im .* x.f[:, :, :, :, :, 1]
                out.f[:, :, :, :, :, 4] .= -im .* x.f[:, :, :, :, :, 2]
            else
                out.f[:, :, :, :, :, 3] .= -x.f[:, :, :, :, :, 1]
                out.f[:, :, :, :, :, 4] .= -x.f[:, :, :, :, :, 2]
            end
        else
            if μ == 1
                out.f[:, :, :, :, :, 3] .= -im .* x.f[:, :, :, :, :, 2]
                out.f[:, :, :, :, :, 4] .= -im .* x.f[:, :, :, :, :, 1]
            elseif μ == 2
                out.f[:, :, :, :, :, 3] .= -x.f[:, :, :, :, :, 2]
                out.f[:, :, :, :, :, 4] .= x.f[:, :, :, :, :, 1]
            elseif μ == 3
                out.f[:, :, :, :, :, 3] .= -im .* x.f[:, :, :, :, :, 1]
                out.f[:, :, :, :, :, 4] .= im .* x.f[:, :, :, :, :, 2]
            else
                out.f[:, :, :, :, :, 3] .= x.f[:, :, :, :, :, 1]
                out.f[:, :, :, :, :, 4] .= x.f[:, :, :, :, :, 2]
            end
        end
        return
    end

    """
    Projects a spinor x in the μ direction onto a halfspinor at a given range of spacetime points

    Parameters
    ----------
    out::ParWilsonFermion
        Output ParWilsonFermion field (ND = 2)
    x::ParWilsonFermion
        Wilson fermion field to project with ND = 4.
    coords_min::SVector{4, Int64}
        (Nx, Ny, Nt, Nz) coordinates to start projection at (should be unpadded).
    coords_max::SVector{4, Int64}
        (Nx, Ny, Nt, Nz) coordinates to start projection at (should be unpadded).
    μ::Integer
        Direction, should be in {1, 2, 3, 4}
    pos::Bool
        If positive or negative projector. True if positive.

    Returns
    -------
    """
    function proj_halfspinor_rg!(out::ParWilsonFermion, x::ParWilsonFermion, coords_min::SVector{4, Int64}, coords_max::SVector{4, Int64}, μ::Integer, plus::Bool)
        xx_min = coords_min[1] + 1
        yy_min = coords_min[2] + 1
        zz_min = coords_min[3] + 1
        tt_min = coords_min[4] + 1

        xx_max = coords_max[1] + 1
        yy_max = coords_max[2] + 1
        zz_max = coords_max[3] + 1
        tt_max = coords_max[4] + 1
        Nc = x.NC

        if plus
            if μ == 1
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1] .- im .* x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 4]          # h0
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2] .- im .* x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 3]          # h1
            elseif μ == 2
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1] .- x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 4]
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2] .+ x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 3]
            elseif μ == 3
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1] .- im .* x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 3]
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2] .+ im .* x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 4]
            else
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1] .- x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 3]
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2] .- x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 4]
            end
        else
            if μ == 1
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1] .+ im .* x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 4]          # h0
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2] .+ im .* x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 3]          # h1
            elseif μ == 2
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1] .+ x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 4]
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2] .- x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 3]
            elseif μ == 3
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1] .+ im .* x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 3]
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2] .- im .* x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 4]
            else
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1] .+ x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 3]
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2] .+ x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 4]
            end
        end
        return
    end

    """
    Reconstructs a projected spinor y to a full spinor x in the μ direction.

    Parameters
    ----------
    out::ParWilsonFermion (ND = 4)
        Reconstructed spinor (two spinor components instead of 4.)
    y::ParWilsonFermion (ND = 2)
        TwoSpinor fermion field to reconstruct.
    coords::SVector{4, Int64}
        (Nx, Ny, Nt, Nz) coordinates to recon at (should be unpadded).
    μ::Integer
        Direction, should be in {1, 2, 3, 4} or {-1, -2, -3, -4}

    Returns
    -------
    ParWilsonFermion
        Reconstructed spinor (two spinor components instead of 4.)
    """
    function recon_halfspinor_rg!(out::ParWilsonFermion, x::ParWilsonFermion, coords_min::SVector{4, Int64}, coords_max::SVector{4, Int64}, μ::Integer, plus::Bool)
        xx_min = coords_min[1] + 1
        yy_min = coords_min[2] + 1
        zz_min = coords_min[3] + 1
        tt_min = coords_min[4] + 1

        xx_max = coords_max[1] + 1
        yy_max = coords_max[2] + 1
        zz_max = coords_max[3] + 1
        tt_max = coords_max[4] + 1
        Nc = x.NC

        out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1]                            # h0
        out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2]                            # h1
        if plus
            if μ == 1
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 3] .= im .* x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2]              # r2
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 4] .= im .* x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1]              # r3
            elseif μ == 2
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 3] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2]
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 4] .= -x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1]
            elseif μ == 3
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 3] .= im .* x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1]
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 4] .= -im .* x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2]
            else
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 3] .= -x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1]
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 4] .= -x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2]
            end
        else
            if μ == 1
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 3] .= -im .* x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2]
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 4] .= -im .* x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1]
            elseif μ == 2
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 3] .= -x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2]
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 4] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1]
            elseif μ == 3
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 3] .= -im .* x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1]
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 4] .= im .* x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2]
            else
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 3] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 1]
                out.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 4] .= x.f[:, xx_min:xx_max, yy_min:yy_max, zz_min:zz_max, tt_min:tt_max, 2]
            end
        end
        return
    end

    """
    Projects a spinor x in the μ direction onto a halfspinor at a given point.

    Parameters
    ----------
    out::ParWilsonFermion
        Output ParWilsonFermion field (ND = 2)
    x::ParWilsonFermion
        Wilson fermion field to project with ND = 4.
    coords::SVector{4, Int64}
        (Nx, Ny, Nt, Nz) coordinates to project at (should be unpadded).
    μ::Integer
        Direction, should be in {1, 2, 3, 4}
    pos::Bool
        If positive or negative projector. True if positive.

    Returns
    -------
    """
    function proj_halfspinor_pt!(out::ParWilsonFermion, x::ParWilsonFermion, coords::SVector{4, Int64}, μ::Integer, plus::Bool)
        xx = coords[1] + 1
        yy = coords[2] + 1
        zz = coords[3] + 1
        tt = coords[4] + 1
        Nc = x.NC

        if plus
            if μ == 1
                for k = 1:Nc
                    out.f[k, xx, yy, zz, tt, 1] = x.f[k, xx, yy, zz, tt, 1] - im * x.f[k, xx, yy, zz, tt, 4]          # h0
                    out.f[k, xx, yy, zz, tt, 2] = x.f[k, xx, yy, zz, tt, 2] - im * x.f[k, xx, yy, zz, tt, 3]          # h1
                end
            elseif μ == 2
                for k = 1:Nc
                    out.f[k, xx, yy, zz, tt, 1] = x.f[k, xx, yy, zz, tt, 1] - x.f[k, xx, yy, zz, tt, 4]
                    out.f[k, xx, yy, zz, tt, 2] = x.f[k, xx, yy, zz, tt, 2] + x.f[k, xx, yy, zz, tt, 3]
                end
            elseif μ == 3
                for k = 1:Nc
                    out.f[k, xx, yy, zz, tt, 1] = x.f[k, xx, yy, zz, tt, 1] - im * x.f[k, xx, yy, zz, tt, 3]
                    out.f[k, xx, yy, zz, tt, 2] = x.f[k, xx, yy, zz, tt, 2] + im * x.f[k, xx, yy, zz, tt, 4]
                end
            else
                for k = 1:Nc
                    out.f[k, xx, yy, zz, tt, 1] = x.f[k, xx, yy, zz, tt, 1] - x.f[k, xx, yy, zz, tt, 3]
                    out.f[k, xx, yy, zz, tt, 2] = x.f[k, xx, yy, zz, tt, 2] - x.f[k, xx, yy, zz, tt, 4]
                end
            end
        else
            if μ == 1
                for k = 1:Nc
                    out.f[k, xx, yy, zz, tt, 1] = x.f[k, xx, yy, zz, tt, 1] + im * x.f[k, xx, yy, zz, tt, 4]          # h0
                    out.f[k, xx, yy, zz, tt, 2] = x.f[k, xx, yy, zz, tt, 2] + im * x.f[k, xx, yy, zz, tt, 3]          # h1
                end
            elseif μ == 2
                for k = 1:Nc
                    out.f[k, xx, yy, zz, tt, 1] = x.f[k, xx, yy, zz, tt, 1] + x.f[k, xx, yy, zz, tt, 4]
                    out.f[k, xx, yy, zz, tt, 2] = x.f[k, xx, yy, zz, tt, 2] - x.f[k, xx, yy, zz, tt, 3]
                end
            elseif μ == 3
                for k = 1:Nc
                    out.f[k, xx, yy, zz, tt, 1] = x.f[k, xx, yy, zz, tt, 1] + im * x.f[k, xx, yy, zz, tt, 3]
                    out.f[k, xx, yy, zz, tt, 2] = x.f[k, xx, yy, zz, tt, 2] - im * x.f[k, xx, yy, zz, tt, 4]
                end
            else
                for k = 1:Nc
                    out.f[k, xx, yy, zz, tt, 1] = x.f[k, xx, yy, zz, tt, 1] + x.f[k, xx, yy, zz, tt, 3]
                    out.f[k, xx, yy, zz, tt, 2] = x.f[k, xx, yy, zz, tt, 2] + x.f[k, xx, yy, zz, tt, 4]
                end
            end
        end
        return
    end

    """
    Reconstructs a projected spinor y to a full spinor x in the μ direction.

    Parameters
    ----------
    out::ParWilsonFermion (ND = 4)
        Reconstructed spinor (two spinor components instead of 4.)
    y::ParWilsonFermion (ND = 2)
        TwoSpinor fermion field to reconstruct.
    coords::SVector{4, Int64}
        (Nx, Ny, Nt, Nz) coordinates to recon at (should be unpadded).
    μ::Integer
        Direction, should be in {1, 2, 3, 4} or {-1, -2, -3, -4}

    Returns
    -------
    ParWilsonFermion
        Reconstructed spinor (two spinor components instead of 4.)
    """
    function recon_halfspinor_pt!(out::ParWilsonFermion, x::ParWilsonFermion, coords::SVector{4, Int64}, μ::Integer, plus::Bool)
        xx = coords[1] + 1
        yy = coords[2] + 1
        zz = coords[3] + 1
        tt = coords[4] + 1
        Nc = x.NC

        for k = 1:Nc
            out.f[k, xx, yy, zz, tt, 1] = x.f[k, xx, yy, zz, tt, 1]                            # h0
            out.f[k, xx, yy, zz, tt, 2] = x.f[k, xx, yy, zz, tt, 2]                            # h1
        end
        if plus
            if μ == 1
                for k = 1:Nc
                    out.f[k, xx, yy, zz, tt, 3] = im * x.f[k, xx, yy, zz, tt, 2]              # r2
                    out.f[k, xx, yy, zz, tt, 4] = im * x.f[k, xx, yy, zz, tt, 1]              # r3
                end
            elseif μ == 2
                for k = 1:Nc
                    out.f[k, xx, yy, zz, tt, 3] = x.f[k, xx, yy, zz, tt, 2]
                    out.f[k, xx, yy, zz, tt, 4] = -x.f[k, xx, yy, zz, tt, 1]
                end
            elseif μ == 3
                for k = 1:Nc
                    out.f[k, xx, yy, zz, tt, 3] = im * x.f[k, xx, yy, zz, tt, 1]
                    out.f[k, xx, yy, zz, tt, 4] = -im * x.f[k, xx, yy, zz, tt, 2]
                end
            else
                for k = 1:Nc
                    out.f[k, xx, yy, zz, tt, 3] = -x.f[k, xx, yy, zz, tt, 1]
                    out.f[k, xx, yy, zz, tt, 4] = -x.f[k, xx, yy, zz, tt, 2]
                end
            end
        else
            if μ == 1
                for k = 1:Nc
                    out.f[k, xx, yy, zz, tt, 3] = -im * x.f[k, xx, yy, zz, tt, 2]
                    out.f[k, xx, yy, zz, tt, 4] = -im * x.f[k, xx, yy, zz, tt, 1]
                end
            elseif μ == 2
                for k = 1:Nc
                    out.f[k, xx, yy, zz, tt, 3] = -x.f[k, xx, yy, zz, tt, 2]
                    out.f[k, xx, yy, zz, tt, 4] = x.f[k, xx, yy, zz, tt, 1]
                end
            elseif μ == 3
                for k = 1:Nc
                    out.f[k, xx, yy, zz, tt, 3] = -im * x.f[k, xx, yy, zz, tt, 1]
                    out.f[k, xx, yy, zz, tt, 4] = im * x.f[k, xx, yy, zz, tt, 2]
                end
            else
                for k = 1:Nc
                    out.f[k, xx, yy, zz, tt, 3] = x.f[k, xx, yy, zz, tt, 1]
                    out.f[k, xx, yy, zz, tt, 4] = x.f[k, xx, yy, zz, tt, 2]
                end
            end
        end
        return
    end

    """
    Projects a spinor x in the μ direction (returns a Dirac spinor).

    Parameters
    ----------
    out::ParWilsonFermion (ND = 4)
        Output ParWilsonFermion field.
    x::ParWilsonFermion (ND = 4)
        Wilson fermion field to project with ND = 4.
    tmp::ParWilsonFermion (ND = 2)
        Halfspinor field for temporary step
    μ::Integer
        Direction, should be in {1, 2, 3, 4}
    pos::Bool
        If positive or negative projector. True if positive.

    Returns
    -------
    ParWilsonFermion
        Projected spinor (ND = 4)
    """
    function project!(out::ParWilsonFermion, x::ParWilsonFermion, tmp::ParWilsonFermion, μ::Integer, plus::Bool)
        proj_halfspinor!(tmp, x, μ, plus)
        recon_halfspinor!(out, tmp, μ, plus)
    end

    ############################################################################
    ########################### Misc extra operations ##########################
    ############################################################################

    """
    Fills in a FermionFields instance with an array of the same shape.
    """
    function fill!(x::FermionFields, mat::Array{ComplexF64,6}; Nd::Integer = 4)
        @threads for it = 1:x.NT
            for α = 1:Nd
                for iz = 1:x.NZ
                    for iy = 1:x.NY
                        for ic = 1:x.NC
                            @simd for ix = 1:x.NX
                                x.f[ic, ix + 1,iy + 1,iz + 1,it + 1, α] = mat[ic, ix, iy, iz, it, α]
                            end
                        end
                    end
                end
            end
        end
        return x
    end

end
