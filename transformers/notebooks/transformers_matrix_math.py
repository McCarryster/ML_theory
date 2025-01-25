import numpy as np
import math

def softmax(matrix):
    e_x = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def layer_norm(x, eps=1e-5):
    # Calculate the mean and variance
    mean = np.mean(x, axis=-1, keepdims=True)            # Mean across the last axis
    variance = np.var(x, axis=-1, keepdims=True)         # Variance across the last axis
    x_normalized = (x - mean) / np.sqrt(variance + eps)  # Normalize
    return x_normalized

# NOTE: There is no back propagation in this code

# Parameters
attention_heads = 8
embedding_len = 512
embedding_per_head = int(512/8)
input_sentence = "I love Defense of the Ancients 2"
input_len = len(input_sentence.split()) # (7 words)

# Initialize weights (W_Q, W_K, W_V)
W_Q_encoder = np.random.rand(attention_heads, embedding_len, embedding_per_head)  # Shape (8, 512, 64)
W_K_encoder = np.random.rand(attention_heads, embedding_len, embedding_per_head)  # Shape (8, 512, 64)
W_V_encoder = np.random.rand(attention_heads, embedding_len, embedding_per_head)  # Shape (8, 512, 64)

# Step 1: Convert words into embeddings
x_embed = np.random.rand(input_len, embedding_len)  # Shape (7, 512)

# Step 2: Add positional encodings to those embeddings elementwise
pos_encodings = np.random.rand(input_len, embedding_len)  # Shape (7, 512)
x_embed_pos_encoded = np.add(x_embed, pos_encodings)  # Shape (7, 512)

# Step 3: Calculate Query Values, Key Values, Value Values (Q, K, V)
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

# STEP 4: Calculate Attention Values
    # Attention(Q, K, V) = Softmax(Q*K.T/√d_k)*V
    # Q*K.T - Unscaled Dot Product similarities between all possible combinations of Queries and Keys for each token
    # d_k - Dimension of Key matrix (It is for scaling purpose)
attention_values = []
for head in range(attention_heads):
    unscaled_dot_product = np.matmul(Q_encoder[head], K_encoder[head].T)         # (7, 7)
    scaled_dot_product = unscaled_dot_product / math.sqrt(x_embed.shape[1])
    softmaxed_scaled_dot_product = softmax(scaled_dot_product)
    attention_output = np.matmul(softmaxed_scaled_dot_product, V_encoder[head])  # Shape (7, 64)
    attention_values.append(attention_output)
concat_attention = np.concatenate(attention_values, axis=1)  # Shape (7, 512)

# STEP 6: Pass concatenated attention values into a Fully Connected Layer
W_O = np.random.rand(512, 512)  # Shape (512, 512)
b_O = np.random.rand(512)       # Shape (512,)
fc_concat_attention = np.matmul(concat_attention, W_O) + b_O  # Shape (7, 512)

# STEP 7: Add Residual Connections and make layer norm
    # The residual connection helps with the vanishing gradient problem which can occur in deep NNs by allowing gradients to flow through the network more easily
fc_concat_attention_res_added = np.add(fc_concat_attention, x_embed_pos_encoded)  # Shape (7, 512)
final_output = layer_norm(fc_concat_attention_res_added)