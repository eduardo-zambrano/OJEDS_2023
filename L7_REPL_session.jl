#] activate .
include("REPL_helper.jl");
using OhMyREPL
using VMLS
using LinearAlgebra

A = [ 0.0 1.0 -2.3 0.1; 1.3 4 -0.1 0; 4.1 -1 0 1.7 ]
size(A)

A[2,3]
Z = [ -1 0 2; -1 2 -3]
Z[5]
A = [ -1 0 1 0; 2 -3 0 1; 0 4 -2 1]
A[1:2,3:4]
A[:,3]
A[2,:]
B = [ 1 -3 ; 2 0; 1 -2]
B[:]
reshape(B,(2,3))
reshape(B,(3,3))

B = [0 2 3];
C = [-1];
D = [2 2 1; 1 3 5];
E = [4, 4];
A = [ B C ; D E]
a = [ [.1, 2.], [4., 5.], [7., 8.]]
A = hcat(a...)
a = [ [1. 2.], [4. 5], [7. 8.]]
A = vcat(a...)
a = [ [.1, 2.], [4., 5.], [7., 8.]]
vcat(a...)

zeros(7,7)

eye(4)
A = [1 -1 2; 0 3 -1]

[A I]
[A ; I]
B = [ 1 2; 3 4]
B + I

diagm(0 => [1,2,3])

diagonal([1,2,3])

rand(2,3)
randn(3,2)

using SparseArrays
M = [ 1, 2 , 2, 1, 3 ,4 ] # row indexes of nonzeros
N = [1, 1, 2, 3, 3, 4] # column indexes of nonzeros
V = [ -1.11, 0.15, -.1, 1.17, -0.3, 0.13]

A = sparse(M, N, V, 4, 5)
B = Array(A)
nnz(A)

D = Diagonal(10:10:50)
D[2,2]
M = diagm(0 => 10:10:50)
S = spidiagm(0 => 10:10:50)
S = spdiagm(0 => 10:10:50)

using BenchmarkTools
N = 2000
D = Diagonal(1:N)
M = diagm(0 =>1:N)
S = spdiagm(0 =>1:N)
x = rand(N)
y = zeros(N)
@btime y = D * x
@btime y = S * x
@btime y = M * x
save_REPL_history("L7_REPL_session.jl")