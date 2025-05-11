# Deep Convolutional Generative Adversarial Network (DCGAN)

This project implements a Deep Convolutional GAN (DCGAN) using Keras and TensorFlow, designed to generate realistic images from random noise. The notebook demonstrates the construction and training of both the generator and discriminator networks, and visualizes the generated outputs.

## Features
- Builds a generator network with dense, reshape, Conv2D, and Conv2DTranspose layers
- Builds a discriminator network with stacked Conv2D and LeakyReLU layers
- Trains the GAN on image data (CIFAR-10 dataset)
- Visualizes generated images during training

## Technologies Used
- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib

## How to Run
1. Open `DeepConvolutionalGAN.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter Notebook.
2. (If using your own dataset, upload it as described in the notebook.)
3. Run all cells in order to build, train, and visualize the GAN.
4. If running locally, first install dependencies:

    ```
    pip install -r requirements.txt
    ```

## Results
- The generator learns to produce images that become increasingly realistic over epochs.
- Example outputs and loss curves are visualized in the notebook.

## References
- [Keras GAN Tutorial](https://keras.io/examples/generative/dcgan_overriding_train_step/)
- [Original GAN Paper (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661)

## Author
- Amirfarhad
