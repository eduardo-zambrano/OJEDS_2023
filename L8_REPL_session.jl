#] activate .
using OhMyREPL
include("REPL_helper.jl")
using VMLS, LinearAlgebra

H = [0 1 -2 1; 2 -1 3 0]
H'
B = 7H
H + B
A = [ 2 3 -1; 0 -1 4]
norm(A)

A[:]
norm(A[:])

A = [-1 0; 2 2]; B = [3 1; -3 2];
norm(A+B), norm(A) + norm(B)
A = [0 2 -1; -2 1 1]
x = [2, 1, -1]
A*x

A = randn(6,4)
Q, R = qr(A);
R
Q
Q = Matrix(Q)

Q*R
Q'*Q

[abs(round(x)) for x in Q'*Q]

save_REPL_history("L8_REPL_session.jl")