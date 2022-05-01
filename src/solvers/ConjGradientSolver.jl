#=
Conjugate Gradient solver for solving Dx=b
=#

module ConjGradientSolverModule 
    using LinearAlgebra
    include("../fermions/ParWilsonFermion.jl")
    using ..ParWilsonFermionModule:ParWilsonFermion
    
    
    function solve_verbose!(D::AbstractMatrix,b::ParWilsonFermion,x::ParWilsonFermion)
        N_matrix=4*x.NC*x.NX*x.NY*x.NZ*x.NT
        
        tolerance = 1e-20

        q=0*x #workspace for mults
        mul!(q,D,x)
        #println(x)
        r = b - q
        p = r

        norm_r = abs( r*r )
        Niterates=0
        for i=1:(4*N_matrix)^2
            Niterates+=1

            r_prev = norm_r
            mul!(q, D,p )
            alpha = r_prev / abs( p*q )
            
            #x needs to be treated specially to be manipulated by pointer...
            for α=1:4
                for it=1:x.NT
                    for iz=1:x.NZ
                        for iy=1:x.NY
                            for ix=1:x.NX
                                @simd for ic=1:x.NC
                                    x[ic,ix,iy,iz,it,α] += alpha*p[ic,ix,iy,iz,it,α]
                                end
                            end
                        end
                    end
                end
            end
            
            r = r - alpha*q
            
            if norm_r < tolerance
                break
            end
            
            norm_r =  abs( r*r )
            beta = norm_r / r_prev
            p = r + beta*p    
            
            println(norm_r)
        end
                
        println("Num Iterations: ",Niterates)
        
    end
    
    
    function solve_verbose!(D::AbstractMatrix,b::AbstractVector{T},x::AbstractVector{T}) where T <: Number
        N_matrix=size(x)[1]
        println(N_matrix)
        tolerance = 1e-10

        q=0*x #workspace for mults
        mul!(q,D,x)
        #println(q)
        r = b - q
        p = r
        r_prev = abs( r'*r )
        Niterates=0
        for i=1:N_matrix
            Niterates+=1
            mul!(q, D,p )
            #println()
            alpha = r_prev / abs( p'*q )
            x .= x + alpha*p
            r = r - alpha*q
            norm_r =  abs( r'*r )
            
            if norm_r < tolerance
                break
            end
            
            beta = norm_r / r_prev
            p = r + beta*p
            r_prev=norm_r
            println(norm_r)
        end
                
        println("Num Iterations: ",Niterates)
        
    end
    
    function solve!(D::AbstractMatrix,b::AbstractVector{T},x::AbstractVector{T}) where T <: Number
        N_matrix=size(x)[1]
        println(N_matrix)
        tolerance = 1e-10

        q=0*x #workspace for mults
        mul!(q,D,x)
        #println(q)
        r = b - q
        p = r
        r_prev = abs( r'*r )
        Niterates=0
        for i=1:N_matrix
            Niterates+=1
            mul!(q, D,p )
            #println()
            alpha = r_prev / abs( p'*q )
            x .= x + alpha*p
            r = r - alpha*q
            norm_r =  abs( r'*r )
            
            if norm_r < tolerance
                break
            end
            
            beta = norm_r / r_prev
            p = r + beta*p
            r_prev=norm_r
            #println(norm_r)
        end
                
        println("Num Iterations: ",Niterates)
        
    end
    
    function solve!(D::AbstractMatrix,b::ParWilsonFermion,x::ParWilsonFermion)
        N_matrix=4*x.NC*x.NX*x.NY*x.NZ*x.NT
        
        tolerance = 1e-20

        q=0*x #workspace for mults
        mul!(q,D,x)
        #println(x)
        r = b - q
        p = r

        norm_r = abs( r*r )
        Niterates=0
        for i=1:(4*N_matrix)^2
            Niterates+=1

            r_prev = norm_r
            mul!(q, D,p )
            alpha = r_prev / abs( p*q )
            x = x + alpha*p
            r = r - alpha*q
            norm_r =  abs( r*r )
            
            if norm_r < tolerance
                break
            end
            
            beta = norm_r / r_prev
            p = r + beta*p
        end
                
    end
    
end
