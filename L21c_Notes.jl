using JuMP, GLPK, Plots
using Random
Random.seed!(598)

# Function to generate a random point within a circle
function random_point_in_circle(center, radius)
    theta = 2 * π * rand()  # Random angle
    r = radius * sqrt(rand())  # Random radius, sqrt to ensure uniform distribution
    x = center[1] + r * cos(theta)
    y = center[2] + r * sin(theta)
    return [x, y]
end

"""
We consider here the case where a perfect linear classifier exists
"""

# Number of points
n = 40

# Generate points for group a
a_center = [10, 10]
a_radius = 5
a_vectors = [random_point_in_circle(a_center, a_radius) for _ in 1:n]

# Generate points for group b
b_center = [20, 20]
b_radius = 5
b_vectors = [random_point_in_circle(b_center, b_radius) for _ in 1:n]


# Extracting x and y coordinates for plotting
a_x = [p[1] for p in a_vectors]
a_y = [p[2] for p in a_vectors]
b_x = [p[1] for p in b_vectors]
b_y = [p[2] for p in b_vectors]

# Plotting
plot(legend=:outerbottomright);
scatter!(a_x, a_y, color=:blue, label="Group A");
scatter!(b_x, b_y, color=:red, label="Group B");
xlabel!("X");
ylabel!("Y");
title!("Randomly Generated Points in Groups A and B");

function linear_classifier(a_vectors, b_vectors)
    # Infer the dimension from the first vector in a_vectors
    d = size(a_vectors[1], 1)

    model = Model(GLPK.Optimizer)

    # Define the variables
    @variable(model, y[1:d])
    @variable(model, beta)
    @variable(model, delta)

    # Set the objective to maximize delta
    @objective(model, Max, delta)

    # Add constraints for a_vectors
    for a in a_vectors
        @constraint(model, a' * y + beta >= 1 + delta)
    end

    # Add constraints for b_vectors
    for b in b_vectors
        @constraint(model, b' * y + beta <=
         1 - delta)
    end

    # Solve the model
    optimize!(model)

    # Get the solution
    optimal_y = value.(y)
    optimal_beta = value(beta)
    optimal_delta = value(delta)

    return optimal_y, optimal_beta, optimal_delta
end

y, beta, delta = linear_classifier(a_vectors, b_vectors)
println("Optimal y: ", y)
println("Optimal beta: ", beta)
println("Optimal delta: ", delta)

# Plot the decision boundary
x_vals = minimum([p[1] for p in a_vectors]):0.1:maximum([p[1] for p in b_vectors]);
y_vals = (-beta .- y[1] .* x_vals) ./ y[2];
plot!(x_vals, y_vals, color=:green, label="Decision Boundary");

title!("Points and Linear Classifier Decision Boundary")


"""
Solve the problem using SVM
"""

using Ipopt

# Function to classify with minimized norm of y
function linear_classifier_min_norm(a_vectors, b_vectors)
    # Infer the dimension from the first vector in a_vectors
    d = size(a_vectors[1], 1)

    model = Model(Ipopt.Optimizer)

    # Define the variables
    @variable(model, y[1:d])
    @variable(model, beta)

    # Set the objective to minimize the norm of y
    @objective(model, Min, sum(y[i]^2 for i in 1:d))

    # Add constraints for a_vectors
    for a in a_vectors
        @constraint(model, a' * y + beta >= 1)
    end

    # Add constraints for b_vectors
    for b in b_vectors
        @constraint(model, b' * y + beta <= -1)
    end

    # Solve the model
    optimize!(model)

    # Get the solution
    optimal_y = value.(y)
    optimal_beta = value(beta)

    return optimal_y, optimal_beta
end

# Solve using the min norm approach
y_min_norm, beta_min_norm = linear_classifier_min_norm(a_vectors, b_vectors)
println("Optimal y (min norm): ", y_min_norm)
println("Optimal beta (min norm): ", beta_min_norm)

# Plot the decision boundary for the min norm solution
y_vals_min_norm = (-beta_min_norm .- y_min_norm[1] .* x_vals) ./ y_min_norm[2]
plot!(x_vals, y_vals_min_norm, color=:purple, label="Min Norm Decision Boundary")

# Display the plot with both decision boundaries
display(plot)

