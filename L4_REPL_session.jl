#activate .
#<BACKSPACE>
using OhMyREPL
include("REPL_helper.jl")

x = [-1.1, 0.0, 3.6, -7.2]
length(x)

# these are not vectors
a = [ 1 2 ]
b = (1, 2)

x[3] = 4
x

x = [-1.1, 0.0, 3.6, -7.2]
y = x
x[3] = 4
y
y[1] = 2.0
y
x
x = [-1.1, 0.0, 3.6, -7.2]
y = copy(x)
x[3] = 4
x
y
x = [-1.1, 0.0, 3.6, -7.2]
y = copy(x)
y[3] = 4
y==x

x = [1, -2]; y = [1, 1, 0];
z = [x; y]
z = [x, y]
z = (x,y)
zeros(3)
ones(5)
[0, 7, 3] + [1, 2, 0]
z = ans
7z
[1.1, -3.7, 0.3] .- 1.4
p_initial = [22.15, 89.32, 56.77]
p_final = [23.05, 87.32, 57.12]
relative_change = (p_final - p_initial) ./p_initial
# Linear combinations
a = [1, 2]; b = [3,4]
α = -0.5; β = 1.5;
c = α*a + β*b
c = αa + βb
d = 2a-7b
#Inner product
x= [-1,2,2]; y = [1,0,-3]
x'*y

using LinearAlgebra
x = [2, -1, 2]
norm(x)
sqrt(x'*x)
x = randn(10)
y = randn(10)
lhs = norm(x+y)
rhs = norm(x) + norm(y)

# Distance
u = [1.8, 2.0, -3.7, 4.7];
v = [0.6, 2.1, 1.9, -1.4];
w = [ 2.0, 1.9, -4.0, 4.6];
norm(u-v)
norm(u-w)
norm(v-w)

coso(x,y) = x'*y / (norm(x)*norm(y))

a = [1,2,-1]; b = [2, 0, -3]
coso(a,b)

save_REPL_history("L4_REPL_session.jl")