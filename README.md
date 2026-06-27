# Sparse Autoencoder MNIST Demo

This example demonstrates a basic sparse autoencoder using TensorFlow/Keras. It trains an autoencoder on a small subset of the MNIST dataset, enforcing sparsity in the hidden (latent) layer's activations using L1 regularization. The output shows how the model reconstructs images and the high percentage of near-zero activations in the latent representation, illustrating the core concept of sparse autoencoders for learning disentangled features.

## Language

`python`

## How to Run

1. Ensure you have Python installed.
2. Install TensorFlow and NumPy: `pip install tensorflow numpy`
3. Run the script: `python main.py`

## Original Article

This example accompanies the Turkish article: [Sparse Autoencoderlar: Yorumlanabilirlik mi, Kırılganlık mı?](https://fatihsoysal.com/blog/sparse-autoencoderlar-yorumlanabilirlik-mi-kirilganlik-mi/).

## License

MIT — see [LICENSE](LICENSE).
