#] activate .
# status
# add GLM
# <BACKSPACE>

using Random, GLM, DataFrames

# Code for explaining the usefulness of Gram - Schmidt for DS

using Random, GLM, DataFrames

# Step 1: Generate the Data
N = 10000

# x_0 is a vector of ones
x = ones(N, 5);
for i in 2:5
    x[:, i] .= randn(N)
end

ϵ = randn(N);
y = x[:,1] .+ x[:,2] .+ 2 .* x[:,3] .+ 3 .* x[:,4] .+ 4 .* x[:,5] .+ ϵ;

# Step 2: OLS Estimation using Julia Package
df = DataFrame(y=y, x1=x[:,2], x2=x[:,3], x3=x[:,4], x4=x[:,5])
ols_model = lm(@formula(y ~ x1 + x2 + x3 + x4), df)

# Step 3: Run Algorithm 3.1 for Regression by successive orthogonalization
function regress_orthogonalize(Y, x)
    N, p = size(x)
    z = copy(x)
    
    for j in 1:p
        for k in 1:(j-1)
            γ = (z[:,j]' * z[:,k]) / (z[:,k]' * z[:,k])
            z[:,j] .-= γ .* z[:,k]
        end
    end
    
    γ_last = (z[:,p]' * Y) / (z[:,p]' * z[:,p])
    return γ_last[1] # Extract the scalar from the 1x1 array
end

γ_4 = regress_orthogonalize(y, x);

println("Estimated γ_4 using successive orthogonalization: ", γ_4)

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