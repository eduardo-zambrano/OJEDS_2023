""" RESOURCES / REFERENCES

# On word embeddings
https://spcman.github.io/getting-to-know-julia/nlp/word-embeddings/

# Illustrating the need for attention
https://lilianweng.github.io/posts/2018-06-24-attention/
"The FBI is chasing a criminal on the run"

# The paper that proposed the transformer architecture
https://arxiv.org/abs/1706.03762

# From simple to more complex models of attention
https://sebastianraschka.com/pdf/lecture-notes/stat453ss21/L19_seq2seq_rnn-transformers__slides.pdf

# Self-attention from scratch
https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html

""" 

# Coding the self-attention mechanism from "Attention is all you need"
using LinearAlgebra

# The input sentence
sentence = "i went to the river bank"

# Create a dictionary with just those words
words = sort(split(replace(sentence, "," => "")))
dc = Dict(word => i for (i, word) in enumerate(words))

# Assign an integer index to each word in the sentence
sentence_words = split(sentence)
sentence_int = [dc[word] for word in sentence_words];
println(sentence_int)

# Create a function to pull out embeddings from an embedding file 
function loado_embeddings(embedding_file)
    local LL, indexed_words, index
    indexed_words = Vector{String}()
    LL = Vector{Vector{Float32}}()
    open(embedding_file) do f
        index = 1
        for line in eachline(f)
            xs = split(line)
            word = xs[1]
            push!(indexed_words, word)
            push!(LL, parse.(Float32, xs[2:end]))
            index += 1
        end
    end
    return reduce(hcat, LL), indexed_words
end

# Load an embedding file from GloVe
# Source: https://nlp.stanford.edu/projects/glove/
embeddings, vocab = loado_embeddings("/Users/ezambran/.julia/datadeps/glove.6B/glove.6B.50d.txt")
vec_size, vocab_size = size(embeddings)

# The function vec_idx returns the index position of a given word in the vocabulary
vec_idx(s) = findfirst(x -> x==s, vocab)

# Example
vec_idx("bank")

# The function vec returns the word vector of the given word.
function vec(s) 
    if vec_idx(s)!=nothing
        embeddings[:, vec_idx(s)]
    end    
end

# Below is the vector for the word “bank”.
vec("bank")

# Some word algebra
norm(vec("human")-vec("child"))
norm(vec("human")-vec("asteroid"))

norm(vec("river")-vec("bank"))
norm(vec("money")-vec("bank"))

# # Get embeddings for words in `sentence_int`
embedding_matrix = hcat([vec(word) for word in sentence_words]...)'
println("Loaded embeddings, each word is represented by a vector with $vec_size features. The vocabulary size is $vocab_size")

# Embed the words in `sentence_int`
embedded_sentence = embedding_matrix[sentence_int, :]

# Get the number of columns (dimensions) from the embedded_sentence matrix
d = size(embedded_sentence, 2)

# These do not have to be equal to one another, but we will make them equal for simplicity
d_q, d_k, d_v = 50, 50, 50

# These matrices would be initialized and then learned. Here, we just initialize them
W_query = rand(d_q, d)
W_key = rand(d_k, d)
W_value = rand(d_v, d)

# Alternatively, let's make them the identity matrix (as in our very simple model of attention)
W_query = I
W_key = I
W_value = I

## Let's compute the context-dependent representation of x_2

# Extract the second row of the embedded_sentence matrix
x_2 = embedded_sentence[2, :]

# Perform matrix-vector multiplication
query_2 = W_query * x_2
key_2 = W_key * x_2
value_2 = W_value * x_2

# Generalize this step and compute this for all of the words
keys = (W_key * embedded_sentence')'
values = (W_value * embedded_sentence')'

# Print the shapes (dimensions) of the resulting matrices
println("keys.shape: ", size(keys))
println("values.shape: ", size(values))

# The 'query' representation of x_2
query_2

# The 'key' representation of x_5
keys[5,:]

# Dot product attention for query_2 and keys[5,:] 
omega_25 = query_2' * keys[5, :];
println(omega_25)

# Dot product attention
omega_2 = query_2' * keys'

# Scaled dot product attention
omega_2_normalized = omega_2 / sqrt(d_k)

# The 'softmax' function
suftmax(x) = exp.(x) ./ sum(exp.(x))

# Attention weights for x_2
attention_weights_2 = suftmax(omega_2_normalized)

context_vector_2 = attention_weights_2 * values

# Let's compute the context vector for all five words using matrix multiplication
x = embedded_sentence

# Perform matrix-matrix multiplication
Q = W_query * x
K = W_key * x
V = W_value * x

ω = Q*K'

# The unnormalized weight on the second word
ω[2,:]'

# Example: the normalized weight on the second word
suftmax(ω[2,:] ./ sqrt(d_k))'

# Example: recall attention_weights_2
attention_weights_2

# The matrix of attention weights
W = hcat(suftmax.([ω[i,:] ./ sqrt(d_k) for i in 1:6])...)'

# The context-dependent representations
A = W * V

context_vector_2
A[2,:]'

"""
The payoff from all this work
"""
# The distance between 'river' and 'bank' in the original embedding
norm(vec("river")-vec("bank"))

# The distance between 'river' and 'bank' in the context-dependent embedding
norm(A[5,:]' - A[6,:]')  
