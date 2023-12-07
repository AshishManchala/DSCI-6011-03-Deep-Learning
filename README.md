# DSCI-6011-03-Deep-Learning

Project: Image Colorization using Deep Learning Models

Overview:
The Image Colorization Deep Learning project aims to automatically add color to grayscale images using various deep learning architectures. The project explores and compares the performance of different models, including ResNet, Unet, GAN (Generative Adversarial Network), and a combination of Unet and GAN (UnetGan). The objective is to enhance grayscale images by predicting and applying realistic and aesthetically pleasing colors.

Project Components:

Dataset:

Utilize a diverse dataset of grayscale images along with their corresponding colored versions for training the models.
Ensure the dataset represents a wide range of scenes and objects to create a robust colorization model.

Deep Learning Models:

ResNet: Employ a deep residual network for image colorization, leveraging its ability to capture intricate features.
Unet: Use a U-shaped convolutional neural network that enables precise localization of features.
GAN: Implement a Generative Adversarial Network to generate realistic colorizations by training a generator against a discriminator.
UnetGan: Combine the strengths of both Unet and GAN to achieve high-quality and accurate colorizations.

Training:

Train each model using the grayscale images from the dataset and their corresponding color images.
Fine-tune hyperparameters and optimize the models to achieve optimal performance.
Consider using transfer learning if applicable, to leverage pre-trained models and accelerate training.

Usage Instructions:

Training:

Run the training script for each model, specifying the dataset path, hyperparameters, and model architecture.
Monitor training progress, adjust parameters as needed, and save the trained models.

Testing/Inference:

Load a pre-trained model or the model obtained after training.
Provide a grayscale input image to the model for colorization.
Save or display the colorized output.

Conclusion:

This project offers a comprehensive exploration of image colorization using deep learning, providing a valuable resource for researchers and students interested in computer vision, deep learning, and image processing. The comparison of different models offers insights into their strengths and weaknesses, guiding future research in the field. The well-documented code and usage instructions ensure accessibility and reproducibility for other researchers and enthusiasts.
