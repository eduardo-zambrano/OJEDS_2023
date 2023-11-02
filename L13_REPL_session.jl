# ] activate .
include("REPL_helper.jl")
using OhMyREPL, LinearAlgebra, VMLS

A = [2 0; -1 1; 0 2]
b = [1,0,-1]
xhat = [1/3, -1/3]
rhat = A*xhat - b
norm(rhat)

x = [1/2,-1/2]
r = A*x - b
norm(r)

inv(A'*A)*A'b
pinv(A)*b

z = [-1.1, 2.3]
(A*z)'*r
(A*z)'*rhat
z = [5.3, -1.2]
(A*z)'*rhat

A = randn(100,20); b = randn(100)
x1 = A\b
x2 = inv(A'*A)*(A'*b)
x3 = pinv(A)*b

Q, R = qr(A)
Q = Matrix(Q);
x4 = R\(Q'*b)


R = [ 0.97 1.86 0.41;
1.23 2.18 0.53;
0.80 1.24 0.62;
1.29 0.98 0.51;
1.10 1.23 0.69;
0.67 0.34 0.54;
0.87 0.26 0.62;
1.10 0.16 0.48;
1.92 0.22 0.71;
1.29 0.12 0.62];
m, n = size(R)

vdes = 1e3 * ones(m)
s = R\vdes
sum(s)

rms(R*s - vdes)

save_REPL_history("L13_REPL_session.jl")