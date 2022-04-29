#=
Conjugate Gradient solver for solving Dx=b
=#

module ConjGradientSolverModule 
    using LinearAlgebra
    include("../fermions/ParWilsonFermion.jl")
    using ..ParWilsonFermionModule:ParWilsonFermion
    
    
    #=function solve!(D::AbstractMatrix,b::ParWilsonFermion,x::ParWilsonFermion)
        #print(D)
        Dt = conj.(D)
        D_Dt = D
        for i=1:4
            for j=1:4
                D_Dt[i,j]=D[i,j]*Dt[i,j]
            end
        end

        tolerance = 1.

        q=0*x #workspace for mults
        mul!(q,D,x)
        println(x)
        r = x - q
        p = r

        norm_r = sqrt( r*r ) + 10*tolerance
        Niterates=0
        while abs(norm_r) > tolerance
            Niterates+=1
            println(abs(norm_r))
            r_prev = norm_r
            mul!(q, Dt,p )
            
            alpha = r_prev / abs(sqrt( q*q ))
            x = x + alpha*p
            mul!(q,D_Dt,p)
            r = r - alpha*q
            beta = abs(sqrt( r*r )) / r_prev
            p = r + beta*p
            norm_r = abs(sqrt( r*r ))
            println(norm_r)
        end
        
        mul!(x,Dt,x)
        
        println("Num Iterations: ",Niterates)
        
    end
    =#
    
    function solve!(D::AbstractMatrix,b::ParWilsonFermion,x::ParWilsonFermion)
        
        tolerance = 1.

        q = 0*x #workspace for mults
        mul!( q,D,x )
        r = b - q
        p = r
        norm_r_squared_prev = r*r 

        Niterates = 0
        while abs( norm_r_squared_prev ) > tolerance
            Niterates += 1
            mul!( q,D,p )
            alpha = norm_r_squared_prev / ( p*q )
            x = x + alpha*p
            r = r - alpha*p
            norm_r_squared_now = r*r
            p = r + ( norm_r_squared_now/norm_r_squared_prev )*p
            norm_r_squared_prev=norm_r_squared_now
            println(abs(norm_r_squared_prev))
        end
                
        println("Num Iterations: ",Niterates)
        
    end
    
end
