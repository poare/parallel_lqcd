#=
WilsonFermion which has extra implementations to speed up computations with WilsonFermion.jl.
=#
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
    @forward (ParWilsonFermion, :wilson) WilsonFermion          # inherit all fields from WilsonFermion?

end
