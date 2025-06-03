import tensorflow as tf
import os
from tensorflow.keras import layers, Model
import numpy as np
from scipy.stats import genpareto
import matplotlib.pyplot as plt

def load_npy_dataset(folder, prefix="", window_size=30, less=False):
    """
    Load dataset from .npy files and prepare windows.

    Args:
        folder: directory containing .npy files
        prefix: e.g., 'machine-1-1_' for SMD
        window_size: length of sliding window
        less: use smaller subset of training data

    Returns:
        train_windows: [N_train, window_size, F]
        test_windows: [N_test, window_size, F]
        labels: [T_test] or [T_test, F]
    """
    train = np.load(os.path.join(folder, f"{prefix}train.npy")).astype(np.float32)
    test = np.load(os.path.join(folder, f"{prefix}test.npy")).astype(np.float32)
    labels = np.load(os.path.join(folder, f"{prefix}labels.npy")).astype(int)

    if less:
        train = train[:int(0.2 * len(train))]

    train_windows, _ = create_windows(train, window_size)
    test_windows, _ = create_windows(test, window_size)

    return train_windows, test_windows, labels

def positional_encoding(length, depth):
    pos = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
    i = tf.range(depth, dtype=tf.float32)[tf.newaxis, :]
    angle_rates = 1 / tf.pow(10000., (2 * (i // 2)) / tf.cast(depth, tf.float32))
    angle_rads = pos * angle_rates
    sines = tf.sin(angle_rads[:, 0::2])
    cosines = tf.cos(angle_rads[:, 1::2])
    return tf.concat([sines, cosines], axis=-1)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training, mask=None):
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))

def build_tranad_model(window_size, num_features, embed_dim=64, num_heads=4, ff_dim=128):
    # Inputs: window [B, W, F] and focus score [B, W, F]
    window_input = layers.Input(shape=(window_size, num_features))
    focus_input = layers.Input(shape=(window_size, num_features))

    # Apply positional encoding
    pe = positional_encoding(window_size, num_features)
    window_pe = window_input + pe
    focus_pe = focus_input + pe

    # Context encoder (self-attention over full sequence)
    context_encoded = TransformerBlock(embed_dim, num_heads, ff_dim)(focus_pe, training=True)

    # Window encoder (masked attention)
    # (simplified without look-ahead mask here for brevity)
    window_encoded = TransformerBlock(embed_dim, num_heads, ff_dim)(window_pe, training=True)

    # Combine window + context
    combined = layers.Concatenate()([window_encoded, context_encoded])
    encoded = layers.Dense(embed_dim, activation='relu')(combined)

    # Two decoders (for adversarial training)
    decoder1 = layers.Dense(num_features, activation='sigmoid', name='decoder1')(encoded)
    decoder2 = layers.Dense(num_features, activation='sigmoid', name='decoder2')(encoded)

    return Model(inputs=[window_input, focus_input], outputs=[decoder1, decoder2])

def create_windows(data, window_size):
    """
    Create sliding windows from multivariate time series.

    Args:
        data: ndarray of shape [T, F] (T time steps, F features)
        window_size: Size of each window

    Returns:
        window_tensor: shape [N, window_size, F]
        focus_tensor: shape [N, window_size, F] (initially zeros)
    """
    T, F = data.shape
    windows = []
    focus_scores = []
    for t in range(window_size - 1, T):
        window = data[t - window_size + 1:t + 1]
        windows.append(window)
        focus_scores.append(np.zeros_like(window))  # zeros at first

    return np.array(windows), np.array(focus_scores)

@tf.function
def train_step(model, optimizer, window_batch, epsilon, epoch):
    # Initial focus is zeros
    focus_batch = tf.zeros_like(window_batch)

    with tf.GradientTape(persistent=True) as tape:
        # Phase 1 — two decoders
        o1, o2 = model([window_batch, focus_batch], training=True)
        focus_score = tf.math.squared_difference(o1, window_batch)  # shape: [B, T, F]

        # Phase 2 — decoder2 re-used
        _, o2_focus = model([window_batch, focus_score], training=True)

        # Compute losses
        n = tf.cast(epoch + 1, tf.float32)
        w_rec = tf.math.pow(epsilon, -n)
        w_adv = 1.0 - w_rec

        # L1: decoder1 (reconstruction + adversarial assist)
        l1 = w_rec * tf.reduce_mean(tf.square(o1 - window_batch)) + \
             w_adv * tf.reduce_mean(tf.square(o2_focus - window_batch))

        # L2: decoder2 (adversarial)
        l2 = w_rec * tf.reduce_mean(tf.square(o2 - window_batch)) - \
             w_adv * tf.reduce_mean(tf.square(o2_focus - window_batch))

        total_loss = l1 + l2

    # Compute gradients
    grads = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return total_loss, l1, l2

def train_tranad(model, X_windows, num_epochs=10, batch_size=64, epsilon=1.01):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    dataset = tf.data.Dataset.from_tensor_slices(X_windows).batch(batch_size).shuffle(1000)

    for epoch in range(num_epochs):
        total_l1 = total_l2 = count = 0
        for step, batch in enumerate(dataset):
            loss, l1, l2 = train_step(model, optimizer, batch, epsilon, epoch)
            total_l1 += l1.numpy()
            total_l2 += l2.numpy()
            count += 1
        print(f"Epoch {epoch+1}: L1 = {total_l1 / count:.5f}, L2 = {total_l2 / count:.5f}")

def infer_tranad(model, test_windows):
    focus_zeros = tf.zeros_like(test_windows)

    # Phase 1: initial reconstruction
    o1, _ = model([test_windows, focus_zeros], training=False)
    focus = tf.math.squared_difference(o1, test_windows)

    # Phase 2: focus-conditioned reconstruction
    _, o2_focus = model([test_windows, focus], training=False)

    # Final anomaly score
    score1 = tf.reduce_mean(tf.square(o1 - test_windows), axis=[1, 2])  # [B]
    score2 = tf.reduce_mean(tf.square(o2_focus - test_windows), axis=[1, 2])  # [B]
    anomaly_scores = 0.5 * (score1 + score2)

    return anomaly_scores.numpy()

def pot_threshold(scores, q=0.98):
    """
    Apply Peak Over Threshold (POT) to get binary anomaly labels.
    Args:
        scores: anomaly scores (1D array)
        q: quantile for thresholding (default: 98th percentile)
    Returns:
        binary_labels: 0 or 1 for each score
        threshold: estimated threshold
    """
    threshold = np.quantile(scores, q)
    excesses = scores[scores > threshold] - threshold
    if len(excesses) < 5:
        return np.zeros_like(scores), threshold

    params = genpareto.fit(excesses)
    pot_thresh = threshold + genpareto.ppf(0.95, *params)
    binary_labels = (scores > pot_thresh).astype(int)
    return binary_labels, pot_thresh


if __name__ == "__main__":
    folder = "processed/SMD"
    prefix = "machine-1-1_"  # Or "" if no prefix
    window_size = 30
    less = False

    # Load from .npy files
    train_windows, test_windows, labels = load_npy_dataset(folder, prefix, window_size, less)

    # Build & train model
    model = build_tranad_model(window_size, num_features=train_windows.shape[2])
    train_tranad(model, train_windows, num_epochs=10)

    # Inference
    scores = infer_tranad(model, tf.convert_to_tensor(test_windows, dtype=tf.float32))
    pred_labels, threshold = pot_threshold(scores)

    # Evaluate if labels exist
    labels = labels[-len(pred_labels):] if labels.ndim == 1 else np.any(labels, axis=1)[-len(pred_labels):]
    tp = np.sum((pred_labels == 1) & (labels == 1))
    precision = tp / max(np.sum(pred_labels), 1)
    recall = tp / max(np.sum(labels), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1)
    print(f"Threshold: {threshold:.4f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(scores, label='Anomaly Score')
    plt.axhline(threshold, color='red', linestyle='--', label='POT Threshold')
    plt.scatter(np.where(pred_labels)[0], scores[pred_labels == 1], color='red', label='Anomalies')
    plt.title("TranAD Anomaly Detection")
    plt.xlabel("Time Window Index")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()
