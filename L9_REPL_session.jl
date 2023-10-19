#] activate .
include("REPL_helper.jl");
using VMLS, OhMyREPL, LinearAlgebra
A = [-3 -4; 4 6; 1 1]
B = [-11 -10 16; 7 8 -11]/9
C = [0 -1 6; 0 1 -4]/2
B*A
C*A

A = [1 -2 3; 0 2 2; -4 -4 -4]
B = inv(A)
B*A
A*B

A = rand(3,3)
inv(A)
Q, R = qr(A);
Q = Matrix(Q)
inv(R)*Q'

diff = inv(A) .- inv(R)*Q'
[abs(round(x)) for x in diff]

function back_subst(R,b)
n = length(b)
x = zeros(n)
for i=n:-1:1
x[i] = (b[i] - R[i,i+1:n]'*x[i+1:n]) / R[i,i]
end
return x
end

R = triu(randn(4,4))
b = rand(4)
x = back_subst(R,b);
norm(R*x - b)

n = 5000
A = randn(n,n); b = randn(n);

@time x1 = A\b
@time x2 = inv(A)*b

save_REPL_history("L9_REPL_session.jl")