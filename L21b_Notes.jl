using JuMP, GLPK, Plots
using Random
Random.seed!(598)

"""
Now, consider the case where a perfect linear classifier does not exist
"""

# Function to generate a random point within a circle
function random_point_in_circle(center, radius)
    theta = 2 * π * rand()  # Random angle
    r = radius * sqrt(rand())  # Random radius, sqrt to ensure uniform distribution
    x = center[1] + r * cos(theta)
    y = center[2] + r * sin(theta)
    return [x, y]
end

# Number of points
n = 40

# Generate points for group a
a_center = [10, 10]
a_radius = 8
a2_vectors = [random_point_in_circle(a_center, a_radius) for _ in 1:n]

# Generate points for group b
b_center = [20, 20]
b_radius = 8
b2_vectors = [random_point_in_circle(b_center, b_radius) for _ in 1:n]


# Extracting x and y coordinates for plotting
a_x = [p[1] for p in a2_vectors]
a_y = [p[2] for p in a2_vectors]
b_x = [p[1] for p in b2_vectors]
b_y = [p[2] for p in b2_vectors]

# Plotting
plot();
scatter!(a_x, a_y, color=:blue, label="Group A");
scatter!(b_x, b_y, color=:red, label="Group B");
xlabel!("X");
ylabel!("Y");
title!("Randomly Generated Points in Groups A and B")

function linear_classifier_2(a_vectors, b_vectors)
    # Infer dimensions
    d = size(a_vectors[1], 1)
    n_1 = length(a_vectors)
    n_2 = length(b_vectors)

    model = Model(GLPK.Optimizer)

    # Define the variables
    @variable(model, y[1:d])
    @variable(model, beta)
    # @variable(model, delta)
    @variable(model, e_1[1:n_1])
    @variable(model, e_2[1:n_2])

    # Set the objective to minimize the sum of the elements of e
    @objective(model, Min,  sum(e_1) + sum(e_2))

    # Add constraints for a_vectors
    for (i,a) in enumerate(a_vectors)
        @constraint(model, a' * y + beta >= 1 - e_1[i])
    end

    # Add constraints for b_vectors
    for (j,b) in enumerate(b_vectors)
        @constraint(model, b' * y + beta <= - (1 - e_2[j]))
    end

    # Solve the model
    optimize!(model)

    # Get the solution
    optimal_y = value.(y)
    optimal_beta = value(beta)
    # optimal_delta = value(delta)
    optimal_e_1 = value.(e_1)
    optimal_e_2 = value.(e_2)


    return optimal_y, optimal_beta, optimal_e_1, optimal_e_2
end

y_2, beta_2, e_1, e_2 = linear_classifier_2(a2_vectors, b2_vectors)
println("Optimal y: ", y_2)
println("Optimal beta: ", beta_2)
println("Optimal e_1: ", e_1)
println("Optimal e_2: ", e_2)


# Plot the decision boundary
x_vals = minimum([p[1] for p in a2_vectors]):0.1:maximum([p[1] for p in b2_vectors]);
y_vals = (-beta_2 .- y_2[1] .* x_vals) ./ y_2[2];
plot!(x_vals, y_vals, color=:green, label="Decision Boundary");

title!("Points and Linear Classifier Decision Boundary")