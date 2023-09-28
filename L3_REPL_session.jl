#]
#activate .
# <BACKSPACE>

include("REPL_helper.jl")
using OhMyREPL

x = true
typeof(x)
y = 1 >2
typeof(y)
typeof(1.0)
typeof(1)
x = 2 ;

x = 2 ; y = 1.0;
x * y
2x - 3y
@show 2x - 3y;
x = "DoctorStrange"
typeof(x)
x ="F"
x = 'F'
typeof(x)
y = "β"
typeof(y)
z = '⚾'
typeof(z)
"y = $z"
"y² = $(z^2)"
"Doctor" * "Strange"
s = "Charlie don't surf"
split(s)
s
replace(s, "surf" => "climb")
strip("      Doctor Strange    ")
x = ("Doctor", "Strange")
# x[1] = "Nurse" -- will not work
y = ["Doctor", "Strange"]
y[1] = "Nurse" # Will work
y
x = [10, 20, 30, 40]
x[end]
x[end-1]
x[1:3]
x[2:end]

"DoctorStrange"[3:end]
actions = ["surf", "ski"]

for action in actions
    println("Charlie don't $action")
end
zz = "Doctor", "Strange"
x_values = 1:5
typeof(x_values)
for x in x_values
    println(x * x)
end
for i in eachindex(x_values)
    println(x_values[i] * x_values[i])
end

# List comprehensions
doubles = [ 2i for i in 1:4 ]
animals = ["dog", "cat", "dodo"];
typeof(animals)

plural = [ animal * "s" for animal in animals]

[ i + j for i in 1:3, j in 4:6 ]
x = 1
x == 2
x != 3
x ≠ 3
1 + 1E-8 ≈ 1
1 + 1E-8 == 1
x = true
!x
true && false
false && true
true || false
false || true

function test(x,y)
           if x < y
               relation = "less than"
           elseif x == y
               relation = "equal to"
           else
               relation = "greater than"
           end
           println("x is ", relation, " y.")
end
test(1,1)
test(2,1)
function test2(x,y)
    return x < y ? "less than" : "not less than"
end
test2(1,1)

function test3(x,y)
           x > y ? "x is greater" : x < y ? "y is greater" : x == y ? "x and y are equal" : "Hi!"
       end

test3(1,1)
test3(2,1)
test3(1,2)       
save_REPL_history("Lecture_3_REPL.jl")