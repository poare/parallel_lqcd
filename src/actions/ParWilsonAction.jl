#=
WilsonFermion which has extra implementations to speed up computations with WilsonFermion.jl.
=#
module ParWilsonFermion

    using ReusePatterns
    using LatticeQCD.AbstractFermion:FermionFields,
        Wx!,Wdagx!,clear!,substitute_fermion!,Dx!,fermion_shift!,fermion_shiftB!,add!,set_wing_fermi!,WdagWx!,apply_periodicity,Ddagx!
    using LatticeQCD.WilsonFermion_module:WilsonFermion,Wx!,Wdagx!,Dx!,Ddagx!

    struct ParWilsonTwoSpinor
        # extra fields that we need here
        
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
        y = ParWilsonTwoSpinor(...)
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
        x = ParWilsonFermion()
        # TODO method stub
        return x
    end

end
