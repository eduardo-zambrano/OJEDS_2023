# Source: https://web.stanford.edu/~boyd/vmls/vmls-julia-companion.pdf
include("REPL_helper.jl");
using OhMyREPL
using VMLS, Plots, Random, Statistics

# Housing Prices 
D = house_sales_data();
area = D["area"];
beds = D["beds"];
condo = D["condo"];
location = D["location"];
price = D["price"];
N = size(price)
X_large = hcat(ones(N), area, max.(area.-1.5, 0), beds, condo, location .== 2, location .== 3, location .== 4 );

θ = X_large \ price

#   15.1 Multi-objective least squares
#   ––––––––––––––––––––––––––––––––––––
# 
#   Let’s write a function that solves the multi-objective least squares
#   problem, with given positive weights. The data are a list (or array) of
#   coefficient matrices (of possibly different heights) As, a matching list of
#   (right-hand side) vectors bs, and the weights, given as an array or list,
#   lambdas.

function mols_solve(As,bs,lambdas)
    k = length(lambdas);
    Atil = vcat([sqrt(lambdas[i])*As[i] for i=1:k]...)
    btil = vcat([sqrt(lambdas[i])*bs[i] for i=1:k]...)
    return Atil \ btil
end

N, p = size(X_large)
npts = 100;
lambdas = 10 .^ linspace(-6,6,npts);
thetas = zeros(p,npts);
for j = 1:npts    
    theta = mols_solve([ X_large, [zeros(p-1) eye(p-1)]],
        [ price, zeros(p-1) ], [1, lambdas[j]])
    thetas[:,j] = theta;
end;

# Plot coefficients
plot(lambdas, thetas', label = ["θ₁" "θ₂" "θ₃" "θ₄" "θ₅" "θ₆" "θ₇" "θ₈"], xscale = :log10)
plot!(xlabel = "lambda", xlim = (1e-6, 1e6))

## Choosing lambda via cross-validation
nfold = div(N,5) # size of first four folds
I = Random.randperm(N);

thetasCV = zeros(p,npts,5);
errorsCV = zeros(npts,5); # prediction errors

for k = 1:5
    if k == 1
        Itrain = I[nfold+1:end];
        Itest = I[1:nfold];
        elseif k == 5
        Itrain = I[1:4*nfold];
        Itest = I[4*nfold+1:end];
    else
        Itrain = I[ [1:(k-1)*nfold ; k*nfold+1 : N]]
        Itest = I[ [(k-1)*nfold+1 ; k*nfold ]];
        end;
    Ntrain = length(Itrain)
    Ntest = length(Itest)
    for j = 1:npts
        theta = mols_solve([ X_large[Itrain,:], [zeros(p-1) eye(p-1)]],
            [ price[Itrain], zeros(p-1) ], [1, lambdas[j]])
        thetasCV[:,j,k] = theta;
        errorsCV[j,k] = rms(X_large[Itest,:] * theta - price[Itest]);
    end;
end;

MSE_r = [mean(errorsCV[i, :]) for i in 1:npts]
plot(lambdas, MSE_r, xscale = :log10);
plot!(xlabel = "lambda", xlim = (1e-6, 1e6))

# Chosen regularization parameter
lambdas[argmin(MSE_r)]

# Remember to 
# 1. save the REPL session.
save_REPL_history("L17_REPL_session.jl")
# 2. GIT commit and GIT push at the end of class.
