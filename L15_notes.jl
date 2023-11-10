# Source: https://web.stanford.edu/~boyd/vmls/vmls-julia-companion.pdf
include("REPL_helper.jl");
using VMLS, LinearAlgebra, Plots, Statistics

#Recall from chapter 12:
R = [0.97 1.86 0.41;
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

# The unconstrained solution
x_unc = R \ vdes

# The matrix and vector that define the constraint
S = [1 0 ; 0 1 ; -1 -1 ]
s = [0 ; 0 ; 1284]

# The modified data
A_R = R*S
vdes_R = vdes - R*s

# The constrained solution
x_R = A_R \ vdes_R
x = S*x_R + s

# Source: https://web.stanford.edu/~boyd/vmls/vmls-julia-companion.pdf
#   Chapter 17
#   ============
# 
#   Constrained least squares applications
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 
#   17.1 Portfolio optimization
#   –––––––––––––––––––––––––––––

# Portfolio value with re-investment, return time series r
cum_value(r) = 10000 * cumprod(1 .+ r)

function port_opt(R,rho)
    T, n = size(R)
    if n == 1
        return [1.0]
    end
    mu = sum(R, dims=1)'/T
    KKT = [ 2*R'*R ones(n) mu; ones(n)' 0 0; mu' 0 0]
    wz1z2 = KKT \ [2*rho*T*mu; 1; rho]
    w = wz1z2[1:n]
    return w
end

R, Rtest = portfolio_data()
T, n = size(R)
Ttest, n = size(Rtest)

ρ = 0.10/250; # Ask for 10% annual return
w = port_opt(R,ρ)
r = R*w; # Portfolio return time series
pf_return = 250*mean(r) #Average annual return

# Plot the in-sample behavior of the porfolio
plot(1:T, cum_value(r), label= "the 10% portfolio", legend=:topleft)

# How do we do out of sample?
rtest = Rtest * w; # Portfolio return time series
pf_return_test = 250 * mean(rtest)
plot(1:Ttest, cum_value(ρ*ones(Ttest)), label= "'steady' 10%")
plot!(1:Ttest, cum_value(rtest), label= "the 10% portfolio underperforms out of sample!", legend=:topleft)

# Problem: Markowitz's curse

save_REPL_history("L15_REPL_session.jl")

# 2. GIT commit and GIT push at the end of class.
