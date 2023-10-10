#] activate .
# status
# add GLM
# <BACKSPACE>

include("REPL_helper.jl")
using OhMyREPL
using Random, GLM, DataFrames
# Step 1: Generate Data
N = 10000
x_0 = ones(N);
x_1 = randn(N);
x_2 = randn(N);
x_3 = randn(N);
x_4 = randn(N);
ϵ = randn(N);
y = x_0  + x_1 + 2x_2 + 3x_3 + 4x_4 + ϵ

# Put the data in a data frame
df = DataFrame(y=y, x1=x_1, x2=x_2, x3=x_3, x4=x_4)
ols_model = lm(@formula(y ~ x1 + x2 + x3 + x4),df)

function regress_orthogonalize(Y, X)
    N, p = size(X)
    Z = copy(X)
    
    for j in 1:p
        for k in 1:(j-1)
            β = (Z[:,j]' * X[:,k]) / (X[:,k]' * X[:,k])
            Z[:,j] .-= β .* X[:,k]
        end
    end
    
    β_last = (Z[:,p]' * Y) / (X[:,p]' * X[:,p])
    return β_last[1] # Extract the scalar from the 1x1 array
end

β_4 = regress_orthogonalize(y,hcat(x_0,x_1,x_2,x_3,x_4))

# Chapter 5 Topics
using LinearAlgebra
a1 = [ 0,0,-1]; a2 = [1, 1,0 ] / sqrt(2); a3 = [1, -1, 0] / sqrt(2);
norm(a1)
norm(a2)
norm(a3)
a1'*a2
a1'*a3
a2'*a3
x = [1, 2, 3]
β1 = a1'*x
β2 = a2'*x
β3 = a3'*x
xexp = β1*a1 + β2*a2 + β3*a3

# Gram - Schmidt
"""
Input: an array [a[1], a[2], ..., a[k] ], containing the k vectors a₁,..., aₖ.
If the vectors are linearly independent, it returns the orthonormal array [q[1], ..., q[k]] 
If the vectors are linearly dependent and the Gram–Schnidt algorithm terminates early in iteration `i`, 
it returns the array [q[1], ..., q[i] ] of length i.
"""
function gram_schmidt(a; tol = 1e-10)
    
    q = []
    for i = 1:length(a)
        qtilde = a[i]
        for j = 1:i-1
            qtilde -= (q[j]'*a[i]) * q[j]
        end
        if norm(qtilde) < tol
            println("Vectors are linearly dependent.")
            return q
        end
        push!(q, qtilde/norm(qtilde))
        end;
    return q
end

a = [ [-1, 1, -1, 1], [-1, 3, -1, 3], [1, 3, 5, 7] ]
q = gram_schmidt(a)

zz=[("norm(q[1]:)",norm(q[1])),
("q[1]'*q[2]:",q[1]'*q[2]),
("q[1]'*q[3]:",q[1]'*q[3]),
("norm(q[2]):",norm(q[2])),
("q[2]'*q[3]:",q[2]'*q[3]),
("norm(q[3]):",norm(q[3]))]

a1exp = (q[1]'*a[1]) * q[1]
a2exp = (q[1]'*a[2]) * q[1] + (q[2]'*a[2]) * q[2]
a3exp = (q[1]'a[3]) * q[1] + (q[2]'*a[3]) * q[2] + (q[3]'*a[3]) * q[3]

b = [ a[1], a[2], 1.3*a[1] + 0.5*a[2] ]

q = gram_schmidt(b)

three_two_vectors = [ [1,1], [1,2], [-1,1] ];
q = gram_schmidt(three_two_vectors)

save_REPL_history("L6_REPL_session.jl")