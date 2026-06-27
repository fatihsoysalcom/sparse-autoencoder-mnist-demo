import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Data Preparation
# Load a small subset of MNIST for a quick demo
print("Loading MNIST dataset...")
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten images: (num_samples, 28, 28) -> (num_samples, 784)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Use a smaller subset for faster execution and demo clarity
x_train = x_train[:10000]
x_test = x_test[:1000]

input_dim = x_train.shape[1] # 784 pixels for MNIST images
latent_dim = 32 # Dimension of the hidden (bottleneck) layer

# 2. Model Definition (Sparse Autoencoder)
# Encoder part of the autoencoder
input_img = keras.Input(shape=(input_dim,))
encoded = layers.Dense(
    latent_dim,
    activation='relu',
    # Key for sparsity: L1 regularization on the hidden layer's activity.
    # This penalty encourages many activations in the 'encoded' layer to be zero.
    activity_regularizer=keras.regularizers.l1(1e-5), # A small L1 penalty
    name="sparse_latent_layer"
)(input_img)

# Decoder part of the autoencoder
decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

# Autoencoder model: maps input to reconstructed output
autoencoder = keras.Model(input_img, decoded)

# Compile the model with an optimizer and a loss function (Mean Squared Error)
autoencoder.compile(optimizer='adam', loss='mse')

print("\nAutoencoder Model Summary:")
autoencoder.summary()

# 3. Training the Sparse Autoencoder
print("\nTraining Sparse Autoencoder (5 epochs)...")
autoencoder.fit(x_train, x_train,
                epochs=5, # Keep epochs low for a quick demo
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                verbose=0) # Suppress verbose output during training for cleaner demo

print("Training complete.")

# 4. Demonstration of Reconstruction and Sparsity
# Create a separate encoder model to extract the sparse latent representations
encoder_model = keras.Model(input_img, encoded)

# Get a few test images to demonstrate
num_images_to_show = 3
test_images = x_test[:num_images_to_show]

# Encode images to get their sparse latent representations
latent_representations = encoder_model.predict(test_images)

# Decode images from the latent representations
reconstructed_images = autoencoder.predict(test_images)

print(f"\nDemonstrating reconstruction and sparsity for {num_images_to_show} test images:")

for i in range(num_images_to_show):
    original_img = test_images[i]
    latent_rep = latent_representations[i]
    reconstructed_img = reconstructed_images[i]

    print(f"\n--- Image {i+1} ---")
    print("Original (first 10 pixels): ", np.round(original_img[:10], 2))
    print("Reconstructed (first 10 pixels):", np.round(reconstructed_img[:10], 2))

    # Calculate sparsity: percentage of activations in the latent layer that are near zero.
    # A higher percentage indicates that the autoencoder has learned a sparse representation.
    sparsity_threshold = 1e-3 # Define what 'near zero' means
    num_zero_activations = np.sum(np.abs(latent_rep) < sparsity_threshold)
    total_activations = latent_rep.size
    sparsity_percentage = (num_zero_activations / total_activations) * 100

    print(f"Sparse Latent Representation (first 10 values): {np.round(latent_rep[:10], 3)}")
    print(f"Sparsity in latent layer: {sparsity_percentage:.2f}% of activations are near zero.")
    # This high percentage of near-zero values is the direct result of the L1 regularization,
    # making the latent representation 'sparse' and potentially more interpretable.