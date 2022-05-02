#=
WilsonFermion which has extra implementations to speed up computations with WilsonFermion.jl.
=#
module ParWilsonFermionModule

    using LinearAlgebra
    using Base.Threads

    # have no idea why using doesn't work but everything breaks
    # that means impoort works but using does not?
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
        NC::Int64                              # Color index
        NX::Int64                              # spatial index
        NY::Int64
        NZ::Int64
        NT::Int64                              # time
        ND::Int64                              # Dirac index 2 or 4
        f::Array{ComplexF64,6}

        γ::Array{ComplexF64,3}
        rplusγ::Array{ComplexF64,3}
        rminusγ::Array{ComplexF64,3}
        hop::Float64                                    # Hopping parameter
        r::Float64                                      # Wilson term
        hopp::Array{ComplexF64,1}                       # TODO what are hopp and hopm? hopping in the + and - directions? # I guess 
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
        return ParWilsonFermion(w.NC, w.NX, w.NY, w.NZ, w.NT, w.ND, w.r, w.hop, w.eps, w.MaxCGstep, w.BoundaryCondition)
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
        # TODO why two extra components here?
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
    Sets the (i1, i2, i3, i4, i5, i6) component of x.f (the fermion field) to v. Note that x.f is 0-indexed in the
    middle and 1-indexed in the color and Dirac components.
    """
    function Base.setindex!(x::ParWilsonFermion, v, i1, i2, i3, i4, i5, i6)
        x.f[i1, i2 + 1, i3 + 1, i4 + 1, i5 + 1, i6] = v # NC, NX, NY, NZ, NT, ND
        # x.f[i2 + 1, i3 + 1, i4 + 1, i5 + 1, i6, i1] = v # NX, NY, NZ, NT, ND, NC
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
        for α=1:a.ND
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
        for α=1:b.ND
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

    function LinearAlgebra.axpy!(a::T,X::WilsonFermion,Y::WilsonFermion) where T <: Number #Y = a*X+Y
        for α=1:X.ND
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

                            # for d1 =  1:4
                            #     for d2 = 1:4
                            #         xout[ic,ix,iy,iz,it,d1] += A[d1,d2]*x[ic,ix,iy,iz,it,d2]

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

                            # for d1 =  1:4
                            #     for d2 = 1:4
                            #         xout[ic,ix,iy,iz,it,d1] += A[d1,d2]*x[ic,ix,iy,iz,it,d2]
                        end
                    end
                end
            end
        end
    end

    # TODO figure out what this function does
    function substitute_fermion!(H, j, x::ParWilsonFermion)
        i = 0
        for ialpha = 1:X.ND
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
            n6 = size(a.f)[6]
            for ialpha=1:ND
                for it=1:NT
                    it1 = it + ifelse(μ ==4,1,0)
                    for iz=1:NZ
                        iz1 = iz + ifelse(μ == 3,1,0)
                        for iy=1:NY
                            iy1 = iy + ifelse(μ == 2,1,0)
                            @simd for ix=1:NX
                                ix1 = ix + ifelse(μ ==1,1,0)

                                b[1,ix,iy,iz,it,ialpha] =   u[μ][1,1,ix,iy,iz,it]*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                            u[μ][1,2,ix,iy,iz,it]*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                            u[μ][1,3,ix,iy,iz,it]*a[3,ix1,iy1,iz1,it1,ialpha]

                                b[2,ix,iy,iz,it,ialpha] =   u[μ][2,1,ix,iy,iz,it]*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                            u[μ][2,2,ix,iy,iz,it]*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                            u[μ][2,3,ix,iy,iz,it]*a[3,ix1,iy1,iz1,it1,ialpha]

                                b[3,ix,iy,iz,it,ialpha] =   u[μ][3,1,ix,iy,iz,it]*a[1,ix1,iy1,iz1,it1,ialpha] +
                                                            u[μ][3,2,ix,iy,iz,it]*a[2,ix1,iy1,iz1,it1,ialpha] +
                                                            u[μ][3,3,ix,iy,iz,it]*a[3,ix1,iy1,iz1,it1,ialpha]

                                # for n1 = 1:3
                                #     for n2 = 1:3
                                #         n1,ix,iy,iz,it,ialpha] += u[μ][n1,n2,ix,iy,iz,it]*a[n2,ix1,iy1,iz1,it1,ialpha]
                                #     end
                                # end

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

                                # for n1 = 1:3
                                #     for n2 = 1:3 
                                #         b[n1,ix,iy,iz,it,ialpha] += conj(u[μ][n1,n2,ix,iy,iz,it])*a[n2,ix1,iy1,iz1,it1,ialpha]
                                #     end
                                # end

                            end
                        end
                    end
                end
            end
        end

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

                                # for n1 = 1:3
                                #     for n2 in 1:3 
                                #         b[n1,ix,iy,iz,it,ialpha] += u[μ][n1,n2,ix,iy,iz,it]*a[n2,ix1,iy1,iz1,it1,ialpha]
                                #     end
                                # end

                            end
                        end
                    end
                end
            end

        elseif μ < 0
            for ialpha =1:ND
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
                                # for n1 = 1:3
                                #     for n2 = 1:3 
                                #         b[n1,ix,iy,iz,it,ialpha] += u[μ][n1,n2,ix,iy,iz,it]*a[n2,ix1,iy1,iz1,it1,ialpha]
                                #     end
                                # end

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
        temp = temps[4]
        temp1 = temps[1]
        temp2 = temps[2]

        clear!(temp)
        set_wing_fermi!(x)              # may want to move this out
        for ν=1:4
            fermion_shift!(temp1,U,ν,x)

            #... Dirac multiplication
            mul!(temp1,view(x.rminusγ,:,:,ν),temp1)

            #
            fermion_shift!(temp2,U,-ν,x)
            mul!(temp2,view(x.rplusγ,:,:,ν),temp2)

            add!(temp,0.5,temp1,0.5,temp2)          # TODO see if we can make add! faster

        end

        clear!(xout)
        add!(xout,1/(2*x.hop),x,-1,temp)

        #display(xout)
        #    exit()
        return
    end

    """
    Evaluates Dx! using the half-spinor projection.
    """
    function Dx_halfspinor!(xout::ParWilsonFermion, U::Array{G,1}, x::ParWilsonFermion, temps::Array{T,1}) where  {T <: FermionFields, G <: GaugeFields}
        temp = temps[4]
        temp1 = temps[1]
        temp2 = temps[2]

        clear!(temp)
        set_wing_fermi!(x)              # may want to move this out

        # TODO function stub

        return
    end

    """
    Multithreaded implementation of Dx!. Lexigraphically breaks the lattice into n_threads sublattices and
    calls Dx_halfspinor! on each one.
    """
    function Dx!(xout::ParWilsonFermion, U::Array{G,1}, x::ParWilsonFermion, temps::Array{T,1}) where  {T <: FermionFields, G <: GaugeFields}
        temp = temps[4]
        temp1 = temps[1]
        temp2 = temps[2]
        n_threads = Threads.nthreads()

        clear!(temp)
        set_wing_fermi!(x)              # may want to move this out

        # TODO function stub

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

            add!(temp,0.5,temp1,0.5,temp2)


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
    Projects a spinor x in the μ direction.

    Parameters
    ----------
    x::ParWilsonFermion
        Wilson fermion field to project with ND = 4.
    μ::Integer
        Direction, should be in {1, 2, 3, 4} or {-1, -2, -3, -4}

    Returns
    -------
    ParWilsonFermion
        Projected spinor (ND = 2)
    """
    function proj(x::ParWilsonFermion, μ::Integer)
        # y = ParWilsonFermion(...)
        # TODO method stub
        return y
    end

    """
    Reconstructs a projected spinor y to a full spinor x in the μ direction.

    Parameters
    ----------
    y::ParWilsonFermion (ND = 2)
        TwoSpinor fermion field to reconstruct.
    μ::Integer
        Direction, should be in {1, 2, 3, 4} or {-1, -2, -3, -4}

    Returns
    -------
    ParWilsonFermion
        Reconstructed spinor (two spinor components instead of 4.)
    """
    function recon(y::ParWilsonFermion, μ::Integer)
        # x = ParWilsonFermion()
        # TODO method stub
        return x
    end

    ############################################################################
    ########################### Misc extra operations ##########################
    ############################################################################

    """
    Fills in a FermionFields instance with an array of the same shape.
    """
    function fill!(x::FermionFields, mat::Array{ComplexF64,6})
        try
            global Nd = x.ND
        catch
            global Nd = 4
        end
        @threads for it = 1:x.NT
            for α = 1:Nd
                for iz = 1:x.NZ
                    for iy = 1:x.NY
                        for ic = 1:x.NC
                            @simd for ix = 1:x.NX
                                x[ic,ix,iy,iz,it,α] = mat[ic,ix,iy,iz,it,α]
                            end
                        end
                    end
                end
            end
        end
        return x
    end

end
