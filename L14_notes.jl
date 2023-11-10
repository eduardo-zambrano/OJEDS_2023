# Source: https://web.stanford.edu/~boyd/vmls/vmls-julia-companion.pdf
include("REPL_helper.jl");
using OhMyREPL
using VMLS, Plots, Random

# Housing Prices - two features plus the constant term
D = house_sales_data()
area = D["area"];
beds = D["beds"];
price = D["price"];
N = length(price)
X = [ ones(N) area beds ];
β = X \ price

scatter(price, X*β, lims = (0,800));
plot!([0, 800], [0, 800], linestyle = :dash);
# make axes equal and add labels
plot!(xlims = (0,800), ylims = (0,800), size = (500,500));
plot!(xlabel = "Actual price", ylabel = "Predicted price (simple model)")


# Housing Prices - Understanding the orthogonalization embedded in least squares (I)

# Preliminaries
area_hat = (ones(N) \ area) * ones(N)


# Begin
r_0 = ones(N);

r_area = area - (r_0 \ area) * r_0

r_area' * r_0

r_beds = beds - (r_0 \ beds) * r_0 - (r_area \ beds) * r_area ;

β_beds = r_beds \ price


# Housing Prices - Understanding the orthogonalization embedded in least squares (II)
beds_hat = [ones(N) area] *([ones(N) area ] \ beds); 
beds_resid = beds - beds_hat;
beds_resid'*beds_hat

β_new = beds_resid \ price


# Housing Prices - two features plus the constant term - Cross-validation
nfold = div(N,5) # size of first four folds
I = Random.randperm(N); # random permutation of numbers 1...N
coeff = zeros(5,3); errors = zeros(5,2);
rms_train = zeros(0)
rms_test = zeros(0)
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
    theta = X[Itrain,:] \ price[Itrain];
    coeff[k,:] = theta;
    append!(rms_train, rms(X[Itrain,:] * theta - price[Itrain]))
    append!(rms_test, rms(X[Itest,:] * theta - price[Itest]))
    end;
coeff # 3 coefficients for the five folds


[rms_train rms_test] # RMS errors for five folds

# Housing Prices - seven features plus the constant term
condo = D["condo"];
location = D["location"];
X_large = hcat(ones(N), area, max.(area.-1.5, 0), beds, condo, location .== 2, location .== 3, location .== 4 );
θ = X_large \ price

rms(X_large*θ - price) # RMS prediction error

scatter(price, X_large*θ , lims = (0,800));
plot!([0, 800], [0, 800], linestyle = :dash);
plot!(xlims = (0,800), ylims = (0,800), size = (500,500));
plot!(xlabel = "Actual price", ylabel = "Predicted price (more complex model")

# Housing Prices - seven features plus the constant term - Cross-validation
models = zeros(8,5); # store 8 coefficients for the 5 models
errors = zeros(2,5); # prediction errors
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
    θ = X_large[Itrain,:] \ price[Itrain];
    errors[1,k] = rms(X_large[Itrain,:] * θ - price[Itrain]);
    errors[2,k] = rms(X_large[Itest,:] * θ - price[Itest]);
    models[:,k] = θ;
end;
# display the eight coefficients for each of the 5 folds 
# models


# Remember to 
# 1. save the REPL session.

save_REPL_history("L14_REPL_session.jl")

# 2. GIT commit and GIT push at the end of class.
