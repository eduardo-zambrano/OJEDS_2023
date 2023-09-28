# Run Julia from your OJEDS_2023 directory
#]
#activate .
#<BACKSPACE>
include("REPL_helper.jl")
using OhMyREPL
x = 3 + 3
ans
ans^2
x = 
3 + 3
ans
using LinearAlgebra
α = 15
α
⛵= 15
#;
#pwd
#ls
#<BACKSPACE>
#?
#sin
#<BACKSPACE>
#;
#ls
#vim README.md
# Stuff happened inside vim
#cat README.md
#git add README.md
#git commit -m "With my comments"
#git push
#?
#sin
#]
#status
#gc
#<BACKSPACE>
using Statistics, Distributions, Plots
n =100
f(x) = x^2
ϵ = randn(n)
plot(f.(ϵ),label="ϵ²")
plot!(ϵ,label="ϵ")
typeof(ϵ)
ϵ[1:5]
n = 100
ϵ² = zeros(n)
for i in 1:n
    ϵ²[i] = randn()^2
end
ϵ²
n = 100
ϵ² = zeros(n)
for i in eachindex(ϵ²)
    ϵ²[i] = randn()^2
end
# you can loop directly over arrays
ϵ_sum = 0.0
m = 5
for ϵ_val in ϵ[1:m]
    ϵ_sum = ϵ_sum + ϵ_val
end
ϵ_mean = ϵ_sum / m
mean(ϵ[1:m])
mean(ϵ[1:m]) ≈ ϵ_mean
function generatedata(n)
    ϵ = zeros(n)
    for i in eachindex(ϵ)
        ϵ[i] = (randn())^2
    end
    return ϵ
end
data = generatedata(10)
plot(data)
function generatedata(n)
    ϵ = randn(n)
    
    for i in eachindex(ϵ)
        ϵ[i] = ϵ[i]^2
    end
    
    return ϵ
end
function generatedata(n)
    ϵ = randn(n)
    return ϵ.^2
end
f(x) = x^2
generatedata(n) = f.(rand(n))
generatedata(n,gen) = gen.(randn(n))
f(x) = x^2

histogram(rand(500))
lp = Laplace()
histogram(rand(lp,500))
save_REPL_history("L2_REPL_session.jl")