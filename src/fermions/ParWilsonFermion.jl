#=
WilsonFermion which has extra implementations to speed up computations with WilsonFermion.jl.
=#
module ParWilsonFermionModule

    using LinearAlgebra

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

    # struct ParWilsonFermion <: FermionFields
    #     serial::WilsonFermion
    # end

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

    struct ParWilsonTwoSpinor <: FermionFields
        NC::Int64
        NX::Int64
        NY::Int64
        NZ::Int64
        NT::Int64
        f::Array{ComplexF64,6}                          # The size of the last component here will be 2

        # TODO the next three fields are specific γ^μ matrices: we either need to scrap them or change the implementation
        # Going to comment them out for the time being
        # γ::Array{ComplexF64,3}
        # rplusγ::Array{ComplexF64,3}
        # rminusγ::Array{ComplexF64,3}

        hop::Float64                                    #Hopping parameter
        r::Float64                                      #Wilson term
        hopp::Array{ComplexF64,1}
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
        return x.NC*x.NX*x.NY*x.NZ*x.NT*4
    end

    """
    Total length of a ParWilsonTwoSpinor.
    """
    function Base.length(x::ParWilsonTwoSpinor)
        return x.NC*x.NX*x.NY*x.NZ*x.NT*2
    end

    """
    Size of a ParWilsonFermion = (Nc, Nx, Ny, Nz, Nt, Nd).
    """
    function Base.size(x::WilsonFermion)
        return (x.NC, x.NX, x.NY, x.NZ, x.NT, 4)
    end

    """
    Size of a ParWilsonTwoSpinor = (Nc, Nx, Ny, Nz, Nt, Nd / 2).
    """
    function Base.size(x::ParWilsonTwoSpinor)
        return (x.NC, x.NX, x.NY, x.NZ, x.NT, 2)
    end

    """
    Iterator over ParWilsonFermion.
    """
    function Base.iterate(x::ParWilsonFermion, i::Int = 1)
        i == length(x.f)+1 && return nothing
        return (x.f[i], i + 1)
    end

    """
    Iterator over ParWilsonTwoSpinor.
    """
    function Base.iterate(x::ParWilsonTwoSpinor, i::Int = 1)
        i == length(x.f)+1 && return nothing
        return (x.f[i], i + 1)
    end

    """
    Constructor for ParWilsonFermion.
    """
    function ParWilsonFermion(w::WilsonFermion)
        return ParWilsonFermion(w.NC, w.NX, w.NY, w.NZ, w.NT, w.r, w.hop, w.eps, w.MaxCGstep, w.BoundaryCondition)
    end

    """
    Constructor for ParWilsonFermion.
    """
    function ParWilsonFermion(NC, NX, NY, NZ, NT, fparam::FermiActionParam, BoundaryCondition)
        r = fparam.r
        hop = fparam.hop
        eps = fparam.eps
        MaxCGstep = fparam.MaxCGstep
        return ParWilsonFermion(NC, NX, NY, NZ, NT, r, hop, eps, MaxCGstep, BoundaryCondition)
    end

    """
    Constructor for ParWilsonTwoSpinor.
    """
    function ParWilsonTwoSpinor(NC, NX, NY, NZ, NT, fparam::FermiActionParam, BoundaryCondition)
        r = fparam.r
        hop = fparam.hop
        eps = fparam.eps
        MaxCGstep = fparam.MaxCGstep
        return ParWilsonTwoSpinor(NC, NX, NY, NZ, NT, r, hop, eps, MaxCGstep, BoundaryCondition)
    end

    """
    Constructor for ParWilsonFermion.
    """
    function ParWilsonFermion(NC, NX, NY, NZ, NT, r, hop, eps, MaxCGstep, BoundaryCondition)
        γ,rplusγ,rminusγ = mk_gamma(r)
        hopp = zeros(ComplexF64,4)
        hopm = zeros(ComplexF64,4)
        hopp .= hop
        hopm .= hop
        Dirac_operator = "Wilson"
        # TODO why two extra components here?
        return ParWilsonFermion(NC,NX,NY,NZ,NT,zeros(ComplexF64,NC,NX+2,NY+2,NZ+2,NT+2,4),
            γ,rplusγ,rminusγ,hop,r,hopp,hopm,eps,Dirac_operator,MaxCGstep,BoundaryCondition)
    end

    """
    Constructor for ParWilsonTwoSpinor.
    """
    function ParWilsonTwoSpinor(NC, NX, NY, NZ, NT, r, hop, eps, MaxCGstep, BoundaryCondition)
        # TODO think about this constructor: do we need γ matrices?
        # γ,rplusγ,rminusγ = mk_gamma(r)
        hopp = zeros(ComplexF64,4)
        hopm = zeros(ComplexF64,4)
        hopp .= hop
        hopm .= hop
        Dirac_operator = "Wilson"
        # return ParWilsonTwoSpinor(NC,NX,NY,NZ,NT,zeros(ComplexF64,NC,NX+2,NY+2,NZ+2,NT+2,4),
        #     γ,rplusγ,rminusγ,hop,r,hopp,hopm,eps,Dirac_operator,MaxCGstep,BoundaryCondition)
        return ParWilsonTwoSpinor(NC,NX,NY,NZ,NT,zeros(ComplexF64,NC,NX+2,NY+2,NZ+2,NT+2,2),
            hop,r,hopp,hopm,eps,Dirac_operator,MaxCGstep,BoundaryCondition)
    end

    """
    Constructor for ParWilsonFermion.
    """
    function ParWilsonFermion(NC,NX,NY,NZ,NT,γ,rplusγ,rminusγ,hop,r,hopp,hopm,eps,fermion,MaxCGstep,BoundaryCondition)
        return ParWilsonFermion(NC,NX,NY,NZ,NT,zeros(ComplexF64,NC,NX+2,NY+2,NZ+2,NT+2,4),
                    γ,rplusγ,rminusγ,hop,r,hopp,hopm,eps,fermion,MaxCGstep,BoundaryCondition)
    end

    """
    Constructor for ParWilsonTwoSpinor.
    """
    function ParWilsonTwoSpinor(NC,NX,NY,NZ,NT,γ,rplusγ,rminusγ,hop,r,hopp,hopm,eps,fermion,MaxCGstep,BoundaryCondition)
        return ParWilsonTwoSpinor(NC,NX,NY,NZ,NT,zeros(ComplexF64,NC,NX+2,NY+2,NZ+2,NT+2,2),
                    hop,r,hopp,hopm,eps,fermion,MaxCGstep,BoundaryCondition)
    end

    """
    Sets the (i1, i2, i3, i4, i5, i6) component of x.f (the fermion field) to v. Note that x.f is 0-indexed in the
    middle and 1-indexed in the color and Dirac components.
    """
    function Base.setindex!(x::ParWilsonFermion, v, i1, i2, i3, i4, i5, i6)
        x.f[i1,i2 + 1,i3 + 1,i4 + 1,i5 + 1,i6] = v
    end

    """
    Sets the (i1, i2, i3, i4, i5, i6) component of x.f (the fermion field) to v.
    """
    function Base.setindex!(x::ParWilsonTwoSpinor, v, i1, i2, i3, i4, i5, i6)
        x.f[i1,i2 + 1,i3 + 1,i4 + 1,i5 + 1,i6] = v
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
    Sets the ii'th component (ii is a vectorized index) of x.f to v.
    """
    function Base.setindex!(x::ParWilsonTwoSpinor, v, ii)
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
        return x.f[i1,i2 .+ 1,i3 .+ 1,i4 .+ 1,i5 .+ 1,i6]
    end

    """
    Gets the (i1, i2, i3, i4, i5, i6) component of x.f (the fermion field).
    """
    function Base.getindex(x::ParWilsonTwoSpinor, i1, i2, i3, i4, i5, i6)
        return x.f[i1,i2 .+ 1,i3 .+ 1,i4 .+ 1,i5 .+ 1,i6]
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

    """
    Gets the ii'th component (ii is a vectorized index) of x.f.
    """
    function Base.getindex(x::ParWilsonTwoSpinor, ii)
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

    We should also give some thought as to which ones of these we'll need for ParWilsonTwoSpinor.
    For the previous set of functions, I reimplemented each function for both ParWilsonFermion and
    for ParWilsonAction, but for these it might be more of a pain than it's worth (there are a lot of
    functions in the WilsonFermion.jl file!)
    =#

    function Base.:*(a::WilsonFermion, b::WilsonFermion)
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

    function Base.:*(a::T,b::WilsonFermion) where T <: Number
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

    function LinearAlgebra.axpy!(a::T,X::WilsonFermion,Y::WilsonFermion) where T <: Number #Y = a*X+Y
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
        for α=1:4
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
    Scalar multiplication of a ParWilsonTwoSpinor.
    """
    function LinearAlgebra.rmul!(a::ParWilsonTwoSpinor, b::T) where T <: Number
        for α=1:2
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

    function Base.similar(x::WilsonFermion)
        return WilsonFermion(x.NC,x.NX,x.NY,x.NZ,x.NT,
                    x.γ,x.rplusγ,x.rminusγ,x.hop,x.r,x.hopp,x.hopm,x.eps,x.Dirac_operator,x.MaxCGstep,x.BoundaryCondition)
    end

    function LinearAlgebra.mul!(xout::WilsonFermion,A::AbstractMatrix,x::WilsonFermion)
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

    function LinearAlgebra.mul!(xout::WilsonFermion,x::WilsonFermion,A::AbstractMatrix)
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

    # TODO figure out what this function does
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

    function substitute_fermion!(H, j, x::ParWilsonTwoSpinor)
        i = 0
        for ialpha = 1:2
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
    ParWilsonFermions and for ParWilsonTwoSpinors
    =#

    function fermion_shift!(b::WilsonFermion,u::Array{T,1},μ::Int,a::WilsonFermion) where T <: SU3GaugeFields
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
            for ialpha=1:4              # when implementing for ParWilsonTwoSpinor, this should be 2
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

    function fermion_shift_gamma!(b::WilsonFermion,u::Array{T,1},μ::Int,a::WilsonFermion) where T <: SU3GaugeFields
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
    ############################# Wilson operators #############################
    ############################################################################

    function Wx!(xout::WilsonFermion,U::Array{G,1},
        x::WilsonFermion,temps::Array{T,1}) where  {T <: FermionFields,G <: GaugeFields}
        temp = temps[4]
        temp1 = temps[1]
        temp2 = temps[2]

        clear!(temp)
        set_wing_fermi!(x)
        for ν=1:4
            fermion_shift!(temp1,U,ν,x)

            #... Dirac multiplication
            mul!(temp1,view(x.rminusγ,:,:,ν),temp1)

            #
            fermion_shift!(temp2,U,-ν,x)
            mul!(temp2,view(x.rplusγ,:,:,ν),temp2)

            add!(temp,x.hopp[ν],temp1,x.hopm[ν],temp2)

        end

        clear!(xout)
        add!(xout,1,x,-1,temp)

        #display(xout)
        #    exit()
        return
    end

    function Wx!(xout::WilsonFermion,U::Array{G,1},
        x::WilsonFermion,temps::Array{T,1},fparam::FermiActionParam_Wilson) where  {T <: FermionFields,G <: GaugeFields}
        Wx!(xout,U,x,temps)
        return
    end




    function Wx!(xout::WilsonFermion,U::Array{G,1},
        x::WilsonFermion,temps::Array{T,1},fparam::FermiActionParam_WilsonClover) where  {T <: FermionFields,G <: GaugeFields}
        Wx!(xout,U,x,temps,fparam.CloverFμν)
        return
    end

    function Wx!(xout::WilsonFermion,U::Array{G,1},
        x::WilsonFermion,temps::Array{T,1},CloverFμν::AbstractArray) where  {T <: FermionFields,G <: GaugeFields}
        temp = temps[4]
        temp1 = temps[1]
        temp2 = temps[2]

        clear!(temp)
        set_wing_fermi!(x)
        for ν=1:4
            fermion_shift!(temp1,U,ν,x)

            #... Dirac multiplication
            mul!(temp1,view(x.rminusγ,:,:,ν),temp1)

            #
            fermion_shift!(temp2,U,-ν,x)
            mul!(temp2,view(x.rplusγ,:,:,ν),temp2)

            add!(temp,x.hopp[ν],temp1,x.hopm[ν],temp2)

        end

        clear!(xout)
        add!(xout,1,x,-1,temp)

        cloverterm!(xout,CloverFμν,x)
        #println( "xout ",xout*xout)



        #display(xout)
        #    exit()
        return
    end

    function Wdagx!(xout::WilsonFermion,U::Array{G,1},
        x::WilsonFermion,temps::Array{T,1},CloverFμν::AbstractArray) where {T <: FermionFields,G <: GaugeFields}
        temp = temps[4]
        temp1 = temps[1]
        temp2 = temps[2]

        clear!(temp)
        x5 = temps[3]

        mul_γ5x!(x5,x)
        set_wing_fermi!(x5)


        for ν=1:4
            fermion_shift!(temp1,U,ν,x5)

            #... Dirac multiplication
            mul!(temp1,view(x.rminusγ,:,:,ν),temp1)

            #
            fermion_shift!(temp2,U,-ν,x5)

            mul!(temp2,view(x.rplusγ,:,:,ν),temp2)

            add!(temp,x.hopp[ν],temp1,x.hopm[ν],temp2)
        end
        clear!(temp1)
        add!(temp1,1,x5,-1,temp)

        cloverterm!(temp1,CloverFμν,x5)

        mul_γ5x!(xout,temp1)
        return
    end

    function Wdagx!(xout::WilsonFermion,U::Array{G,1},
        x::WilsonFermion,temps::Array{T,1},fparam::FermiActionParam_WilsonClover) where {T <: FermionFields,G <: GaugeFields}
        Wdagx!(xout,U,x,temps,fparam.CloverFμν)
        return

    end


    function Wdagx!(xout::WilsonFermion,U::Array{G,1},
        x::WilsonFermion,temps::Array{T,1}) where  {T <: FermionFields,G <: GaugeFields}
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

            add!(temp,x.hopp[ν],temp1,x.hopm[ν],temp2)


        end

        clear!(xout)
        add!(xout,1,x,-1,temp)

        #display(xout)
        #    exit()
        return
    end

    function Wdagx!(xout::WilsonFermion,U::Array{G,1},
        x::WilsonFermion,temps::Array{T,1},fparam::FermiActionParam_Wilson) where {T <: FermionFields,G <: GaugeFields}
        Wdagx!(xout,U,x,temps)
        return
    end

    function Dx!(xout::WilsonFermion,U::Array{G,1},
        x::WilsonFermion,temps::Array{T,1}) where  {T <: FermionFields,G <: GaugeFields}
        temp = temps[4]
        temp1 = temps[1]
        temp2 = temps[2]

        clear!(temp)
        set_wing_fermi!(x)
        for ν=1:4
            fermion_shift!(temp1,U,ν,x)

            #... Dirac multiplication
            mul!(temp1,view(x.rminusγ,:,:,ν),temp1)

            #
            fermion_shift!(temp2,U,-ν,x)
            mul!(temp2,view(x.rplusγ,:,:,ν),temp2)

            add!(temp,0.5,temp1,0.5,temp2)

        end

        clear!(xout)
        add!(xout,1/(2*x.hop),x,-1,temp)

        #display(xout)
        #    exit()
        return
    end

    function Ddagx!(xout::WilsonFermion,U::Array{G,1},
        x::WilsonFermion,temps::Array{T,1}) where  {T <: FermionFields,G <: GaugeFields}
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
    function mul_γ5x!(y::ParWilsonFermion,x::ParWilsonFermion)
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
        Wilson fermion field to project.
    μ::Integer
        Direction, should be in {1, 2, 3, 4} or {-1, -2, -3, -4}

    Returns
    -------
    ParWilsonTwoSpinor
        Projected spinor (two spinor components instead of 4.)
    """
    function proj(x::ParWilsonFermion, μ::Integer)
        # y = ParWilsonTwoSpinor(...)
        # TODO method stub
        return y
    end

    """
    Reconstructs a projected spinor y to a full spinor x in the μ direction.

    Parameters
    ----------
    y::ParWilsonTwoSpinor
        TwoSpinor fermion field to reconstruct.
    μ::Integer
        Direction, should be in {1, 2, 3, 4} or {-1, -2, -3, -4}

    Returns
    -------
    ParWilsonFermion
        Reconstructed spinor (two spinor components instead of 4.)
    """
    function recon(y::ParWilsonTwoSpinor, μ::Integer)
        # x = ParWilsonFermion()
        # TODO method stub
        return x
    end

end
