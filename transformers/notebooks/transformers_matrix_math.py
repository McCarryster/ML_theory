import numpy as np
import math


# There is a lot of boilerplate code out there now, but this implementation gives a good step-by-step understanding of the architecture
# NOTE: There is no back propagation in this code

def softmax(matrix):
    e_x = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)
def layer_norm(x, eps=1e-5):
    # Calculate the mean and variance
    mean = np.mean(x, axis=-1, keepdims=True)            # Mean across the last axis
    variance = np.var(x, axis=-1, keepdims=True)         # Variance across the last axis
    x_normalized = (x - mean) / np.sqrt(variance + eps)  # Normalize
    return x_normalized

# Parameters
attention_heads = 8
embedding_len = 512
embedding_per_head = int(512/8)
input_sentence = "I love defense of the ancients 2"
output_sentence = "Я люблю защиту древних 2"
input_len = len(input_sentence.split()) # (7 words)
output_len = len(output_sentence.split()) # (5 words)

# Initialize weights (W_Q, W_K, W_V)
W_Q_encoder = np.random.rand(attention_heads, embedding_len, embedding_per_head)  # Shape (8, 512, 64)
W_K_encoder = np.random.rand(attention_heads, embedding_len, embedding_per_head)  # Shape (8, 512, 64)
W_V_encoder = np.random.rand(attention_heads, embedding_len, embedding_per_head)  # Shape (8, 512, 64)

# STEP 1: Convert words into embeddings
x_embed = np.random.rand(input_len, embedding_len)  # Shape (7, 512)

# STEP 2: Add positional encodings to those embeddings elementwise
input_pos_encodings = np.random.rand(input_len, embedding_len)  # Shape (7, 512)
x_embed_pos_encoded = np.add(x_embed, input_pos_encodings)  # Shape (7, 512)

# STEP 3: Calculate Query Values, Key Values, Value Values (Q, K, V)
    # NOTE: V_encoder can have different dimensions
    # The Q and K are projected to the same dimension (d_k) for the attention calculation
    # While the V can be projected to a different dimension (d_v) to enrich the output representation
    # This design choice enhances the model's ability to learn and perform various tasks effectively
Q_encoder = np.zeros((attention_heads, input_len, embedding_per_head))  # Shape (8, 7, 64)
K_encoder = np.zeros((attention_heads, input_len, embedding_per_head))  # Shape (8, 7, 64)
V_encoder = np.zeros((attention_heads, input_len, embedding_per_head))  # Shape (8, 7, 64)
for head in range(attention_heads):
    Q_encoder[head] = np.matmul(x_embed_pos_encoded, W_Q_encoder[head])  # Shape (7, 64)
    K_encoder[head] = np.matmul(x_embed_pos_encoded, W_K_encoder[head])  # Shape (7, 64)
    V_encoder[head] = np.matmul(x_embed_pos_encoded, W_V_encoder[head])  # Shape (7, 64)

# STEP 4: Calculate Attention Values for Encoder
    # Attention(Q, K, V) = Softmax(Q*K.T/√d_k)*V
    # Q*K.T - Unscaled Dot Product similarities between all possible combinations of Queries and Keys for each token
    # d_k - Dimension of Key matrix (It is for scaling purpose)
attention_values = []
for head in range(attention_heads):
    unscaled_dot_product = np.matmul(Q_encoder[head], K_encoder[head].T)         # (7, 7)
    scaled_dot_product = unscaled_dot_product / math.sqrt(embedding_per_head)
    softmaxed_scaled_dot_product = softmax(scaled_dot_product)
    attention_output = np.matmul(softmaxed_scaled_dot_product, V_encoder[head])  # Shape (7, 64)
    attention_values.append(attention_output)
concat_attention = np.concatenate(attention_values, axis=1)  # Shape (7, 512)

# STEP 5: Pass concatenated attention values into a Linear layer
W_O = np.random.rand(512, 512)
b_O = np.random.rand(512)
fc_concat_attention = np.matmul(concat_attention, W_O) + b_O  # Shape (7, 512)

# STEP 6: Add Residual Connections and make layer norm
    # The residual connection helps with the vanishing gradient problem which can occur in deep NNs by allowing gradients to flow through the network more easily
fc_concat_attention_res_added = np.add(fc_concat_attention, x_embed_pos_encoded)  # Shape (7, 512)
add_and_norm_output = layer_norm(fc_concat_attention_res_added)

# STEP 7: Feed Forward (2 hidden layers, 512 and 2048)
W_1 = np.random.rand(512, 2048)
b_1 = np.random.rand(2048)
W_2 = np.random.rand(2048, 512)
b_2 = np.random.rand(512)
ff_1 = np.matmul(add_and_norm_output, W_1) + b_1 # (7,2048)
# Relu here
ff_2 = np.matmul(ff_1, W_2) + b_2 # (7,512)

# STEP 8: Add Residual Connections and make layer norm again
ff_2_residual_added = np.add(ff_2, add_and_norm_output)  # Shape (7, 512)
encoder_output = layer_norm(ff_2_residual_added)


# -----------------------------------------------------------
# NOW WE ARE DONE WITH THE ENCODER SECTION OF THE TRANSFORMER
# -----------------------------------------------------------


# Initialize weights (W_Q, W_K, W_V)
W_Q_decoder = np.random.rand(attention_heads, embedding_len, embedding_per_head)  # Shape (8, 512, 64)
W_K_decoder = np.random.rand(attention_heads, embedding_len, embedding_per_head)  # Shape (8, 512, 64)
W_V_decoder = np.random.rand(attention_heads, embedding_len, embedding_per_head)  # Shape (8, 512, 64)

# Step 9: Convert words into embeddings
y_embed = np.random.rand(output_len, embedding_len)  # Shape (5, 512)

# Step 10: Add positional encodings to those embeddings elementwise
output_pos_encodings = np.random.rand(output_len, embedding_len)  # Shape (5, 512)
y_embed_pos_encoded = np.add(y_embed, output_pos_encodings)  # Shape (5, 512)

# Step 11: Masking for the output sequence (to prevent attending to future tokens)
mask = np.triu(np.ones((output_len, output_len)), k=1).astype('bool')

# Step 12: Calculate Query Values, Key Values, Value Values (Q, K, V)
    # NOTE: V_encoder can have different dimensions
    # The Q and K are projected to the same dimension (d_k) for the attention calculation
    # While the V can be projected to a different dimension (d_v) to enrich the output representation
    # This design choice enhances the model's ability to learn and perform various tasks effectively
Q_decoder = np.zeros((attention_heads, output_len, embedding_per_head))  # Shape (8, 5, 64)
K_decoder = np.zeros((attention_heads, output_len, embedding_per_head))  # Shape (8, 5, 64)
V_decoder = np.zeros((attention_heads, output_len, embedding_per_head))  # Shape (8, 5, 64)
for head in range(attention_heads):
    Q_decoder[head] = np.matmul(y_embed_pos_encoded, W_Q_decoder[head])  # Shape (5, 64)
    K_decoder[head] = np.matmul(y_embed_pos_encoded, W_K_decoder[head])  # Shape (5, 64)
    V_decoder[head] = np.matmul(y_embed_pos_encoded, W_V_decoder[head])  # Shape (5, 64)

# STEP 13: Calculate Attention Values for Decoder
    # Attention(Q, K, V) = Softmax(Q*K.T/√d_k)*V
    # Q*K.T - Unscaled Dot Product similarities between all possible combinations of Queries and Keys for each token
    # d_k - Dimension of Key matrix (It is for scaling purpose)
decoder_attention_values = []
for head in range(attention_heads):
    unscaled_dot_product = np.matmul(Q_decoder[head], K_decoder[head].T)         # (5, 5)
    scaled_dot_product = unscaled_dot_product / math.sqrt(embedding_per_head)
    if mask is not None:
        scaled_dot_product = np.where(mask, -np.inf, scaled_dot_product)         # Aplies -infinity to future tokens (masking)
    softmaxed_scaled_dot_product = softmax(scaled_dot_product)
    attention_output = np.matmul(softmaxed_scaled_dot_product, V_decoder[head])  # Shape (5, 64)
    decoder_attention_values.append(attention_output)
concat_decoder_attention = np.concatenate(decoder_attention_values, axis=1)      # Shape (5, 512)

# STEP 14: Pass concatenated attention values into a Linear layer
W_O_decoder = np.random.rand(512, 512)
b_O_decoder = np.random.rand(512)
fc_concat_decoder_attention = np.matmul(concat_decoder_attention, W_O_decoder) + b_O_decoder  # Shape (5, 512)

# STEP 15: Add Residual Connections and make layer norm
    # The residual connection helps with the vanishing gradient problem which can occur in deep NNs by allowing gradients to flow through the network more easily
fc_concat_decoder_attention_res_added = np.add(fc_concat_decoder_attention, y_embed_pos_encoded)  # Shape (5, 512)
decoder_multi_head_attention_output = layer_norm(fc_concat_decoder_attention_res_added)


# -----------------------------------------------------
# NOW WE ARE DONE WITH THE DECODER MULTI HEAD ATTENTION
# -----------------------------------------------------


# Initialize weights (W_Q, W_K, W_V)
W_Q_encoder_decoder = np.random.rand(attention_heads, embedding_len, embedding_per_head)  # Shape (8, 512, 64)
W_K_encoder_decoder = np.random.rand(attention_heads, embedding_len, embedding_per_head)  # Shape (8, 512, 64)
W_V_encoder_decoder = np.random.rand(attention_heads, embedding_len, embedding_per_head)  # Shape (8, 512, 64)

# Step 16: Calculate Query Values, Key Values, Value Values (Q, K, V)
    # 1. NOTE: Use Encoder output as 1st argument for matmul for K and V
    # 2. NOTE: V_encoder can have different dimensions
    # The Q and K are projected to the same dimension (d_k) for the attention calculation
    # While the V can be projected to a different dimension (d_v) to enrich the output representation
    # This design choice enhances the model's ability to learn and perform various tasks effectively
Q_encoder_decoder = np.zeros((attention_heads, output_len, embedding_per_head))  # Shape (8, 5, 64)
K_encoder_decoder = np.zeros((attention_heads, input_len, embedding_per_head))  # Shape (8, 7, 64)
V_encoder_decoder = np.zeros((attention_heads, input_len, embedding_per_head))  # Shape (8, 7, 64)
for head in range(attention_heads):
    # Encoder output = (7, 512)
    # Decoder Multi Head Attention output = (5, 512)
    Q_encoder_decoder[head] = np.matmul(decoder_multi_head_attention_output, W_Q_encoder_decoder[head])  # (5, 512)X(512,64) = (5,64)
    K_encoder_decoder[head] = np.matmul(encoder_output, W_K_encoder_decoder[head])  # (7,512)X(512,64) = (7,64)
    V_encoder_decoder[head] = np.matmul(encoder_output, W_V_encoder_decoder[head])  # (5,512)X(512,64) = (7,64)

# Step 17: Multi-head Attention for Encoder Decoder
    # Attention(Q, K, V) = Softmax(Q*K.T/√d_k)*V
    # Q*K.T - Unscaled Dot Product similarities between all possible combinations of Queries and Keys for each token
    # d_k - Dimension of Key matrix (It is for scaling purpose)
encoder_decoder_attention_values = []
for head in range(attention_heads):
    unscaled_dot_product = np.matmul(Q_encoder_decoder[head], K_encoder_decoder[head].T)    # (5,64)X(64,7) = (5,7)
    scaled_dot_product = unscaled_dot_product / math.sqrt(embedding_per_head)
    softmaxed_scaled_dot_product = softmax(scaled_dot_product)
    attention_output = np.matmul(softmaxed_scaled_dot_product, V_encoder_decoder[head])     # (5,7)X(7,64) = (5,64)
    encoder_decoder_attention_values.append(attention_output)
concat_encoder_decoder_attention = np.concatenate(encoder_decoder_attention_values, axis=1) # Shape (5, 512)

# STEP 18: Pass concatenated attention values into a Linear layer
W_O_encoder_decoder = np.random.rand(512, 512)
b_O_encoder_decoder = np.random.rand(512)
fc_concat_encoder_decoder_attention = np.matmul(concat_encoder_decoder_attention, W_O_encoder_decoder) + b_O_encoder_decoder  # (5,512)X(512,512)+(512) = (5,512)

# STEP 19: Add Residual Connections and make layer norm
    # The residual connection helps with the vanishing gradient problem which can occur in deep NNs by allowing gradients to flow through the network more easily
fc_concat_encoder_decoder_attention_res_added = np.add(fc_concat_encoder_decoder_attention, decoder_multi_head_attention_output)  # Shape (5, 512)
encoder_decoder_multi_head_attention_output = layer_norm(fc_concat_encoder_decoder_attention_res_added)

# STEP 20: Feed Forward (2 hidden layers, 512 and 2048)
W_1_encoder_decoder = np.random.rand(512, 2048)
b_1_encoder_decoder = np.random.rand(2048)
W_2_encoder_decoder = np.random.rand(2048, 512)
b_2_encoder_decoder = np.random.rand(512)
ff_1_encoder_decoder = np.matmul(encoder_decoder_multi_head_attention_output, W_1_encoder_decoder) + b_1_encoder_decoder # (5,512)X(512,2048)+2048 = (5,2048)
# Relu here
ff_2_encoder_decoder = np.matmul(ff_1_encoder_decoder, W_2_encoder_decoder) + b_2_encoder_decoder # (5,2048)X(2048,512)+512 = (5,512)

# STEP 21: Add Residual Connections and make layer norm
    # The residual connection helps with the vanishing gradient problem which can occur in deep NNs by allowing gradients to flow through the network more easily
ff_2_encoder_decoder_attention_res_added = np.add(ff_2_encoder_decoder, encoder_decoder_multi_head_attention_output)  # Shape (5, 512)
encoder_decoder_output = layer_norm(fc_concat_encoder_decoder_attention_res_added) # Shape (5, 512)

# STEP 22: Pass encoder_decoder_output into a Linear layer
    # In real implementation it will be like that (embedding_size, vocabulary_size)
W_O_final = np.random.rand(512, output_len)
b_O_final = np.random.rand(output_len)
linear_output = np.matmul(encoder_decoder_output, W_O_final) + b_O_final  # Shape (5, 5)

# STEP 23. Apply softmax to get output word
softmax_output = softmax(linear_output)
print(softmax_output)