import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

def self_attention(Q, K, V):
    scores = np.dot(Q, K.T)
    weights = softmax(scores)
    out = np.dot(weights, V)
    return out, weights

def multi_head_attention(x, num_heads=2):
    d_model = x.shape[-1]
    depth = d_model // num_heads
    outputs = []
    weights_all = []

    for i in range(num_heads):
        Q = x[:, i*depth:(i+1)*depth]
        K = x[:, i*depth:(i+1)*depth]
        V = x[:, i*depth:(i+1)*depth]
        out, weights = self_attention(Q, K, V)
        outputs.append(out)
        weights_all.append(weights)

    combined = np.concatenate(outputs, axis=-1)
    return combined, weights_all

def feed_forward(x, hidden_size=8):
    return np.maximum(0, x @ np.random.randn(x.shape[-1], hidden_size)) @ np.random.randn(hidden_size, x.shape[-1])

def transformer_encoder(x, num_heads=2):
    attn_out, attn_weights = multi_head_attention(x, num_heads)
    x = layer_norm(x + attn_out)

    ff_out = feed_forward(x)
    x = layer_norm(x + ff_out)

    return x, attn_weights

x = np.random.rand(4, 8)

output, weights = transformer_encoder(x, num_heads=2)

print("Transformer encoder çıktısı (vektörler):")
print(output)

sns.heatmap(weights[0], annot=True, cmap="YlGnBu")
plt.title("Attention Head 1")
plt.show()