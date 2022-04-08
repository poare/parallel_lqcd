

module ParWilsonFermion

    using ReusePatterns
    using LatticeQCD.AbstractFermion:FermionFields,
        Wx!,Wdagx!,clear!,substitute_fermion!,Dx!,fermion_shift!,fermion_shiftB!,add!,set_wing_fermi!,WdagWx!,apply_periodicity,Ddagx!
    using LatticeQCD.WilsonFermion_module:WilsonFermion,Wx!,Wdagx!,Dx!,Ddagx!

    # struct ParWilsonFermion <: WilsonFermion
    #     ...fields
    # end
    struct ParWilsonFermion
        wilson::WilsonFermion
        # extra fields that we need here
    end

    struct ParWilsonTwoSpinor <: FermionFields
        NC::Int64
        NX::Int64
        NY::Int64
        NZ::Int64
        NT::Int64
        f::Array{ComplexF64,6}

        γ::Array{ComplexF64,3}
        rplusγ::Array{ComplexF64,3}
        rminusγ::Array{ComplexF64,3}
        hop::Float64 #Hopping parameter
        r::Float64 #Wilson term
        hopp::Array{ComplexF64,1}
        hopm::Array{ComplexF64,1}
        eps::Float64
        Dirac_operator::String
        MaxCGstep::Int64
        BoundaryCondition::Array{Int8,1}
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
        x = ParWilsonFermion()
        # TODO method stub
        return x
    end

end
