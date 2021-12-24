# Quick Draw GAN

In this project I implemented and trained a DCGAN model on the [Google Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset). The GAN is able to generate 28x28 images resembling drawings made by people.

## Table of Contents
- [Model](#model)
- [Technologies](#technologies)
- [Sources](#sources)

## Model
The GAN is made up of two neural networks, one called the discriminator and the other called the generator. The discriminator classifies images fed into it as either real or fake. The generator generates images when fed vectors from the latent space. A summary of the layers of these models is as follows:

---

Model: generator
|Layer (type) | Output Shape | Param #|
|:---------------|:---------------|:---------------|
|input_2 (InputLayer)| (None, 100) | 0 |
|dense_1 (Dense)|              (None, 8192)|              819200| 
|reshape (Reshape)|            (None, 4, 4, 512)|         0|         
|conv2d_transpose (Conv2DTranspose)| (None, 8, 8, 256)|2097408|
|batch_normalization_3 (BatchNormalization)| (None, 8, 8, 256)|         1024      |
|leaky_re_lu_4 (LeakyReLU)|    (None, 8, 8, 256)|         0|         
|conv2d_transpose_1| (None, 16, 16, 128)|       524416    |
|batch_normalization_4 (BatchNormalization)| (None, 16, 16, 128)|       512       |
|leaky_re_lu_5 (LeakyReLU)|    (None, 16, 16, 128)|       0         |
|conv2d_transpose_2 (Conv2DTranspose)| (None, 32, 32, 64)|        131136    |
|batch_normalization_5 (BatchNormalization)| (None, 32, 32, 64)|        256       |
|leaky_re_lu_6 (LeakyReLU)|    (None, 32, 32, 64)|        0         |
|conv2d_transpose_3 (Conv2DTranspose)| (None, 32, 32, 1)|         1025      |
|activation_1 (Activation)|    (None, 32, 32, 1)|         0         |
|cropping2d (Cropping2D)|      (None, 28, 28, 1)|         0         |

---

Model: discriminator
|Layer (type) | Output Shape | Param #|
|:---------------|:---------------|:---------------|
|input_1 (InputLayer)|         (None, 28, 28, 1) |      0         |
|zero_padding2d (ZeroPadding2D)| (None, 32, 32, 1)|         0         |
|conv2d (Conv2D)|              (None, 16, 16, 64)|        1088      |
|batch_normalization (BatchNormalization)| (None, 16, 16, 64)|        256       |
|leaky_re_lu_1 (LeakyReLU)|    (None, 16, 16, 64)|        0         |
|conv2d_1 (Conv2D)|            (None, 8, 8, 128)|         131200    |
|batch_normalization_1 (BatchNormalization)| (None, 8, 8, 128)|         512       |
|leaky_re_lu_2 (LeakyReLU)|    (None, 8, 8, 128)|         0         |
|conv2d_2 (Conv2D)|            (None, 4, 4, 256)|         524544    |
|batch_normalization_2 (BatchNormalization)| (None, 4, 4, 256)|         1024      |
|leaky_re_lu_3 (LeakyReLU)|    (None, 4, 4, 256)|         0         |
|flatten (Flatten)|            (None, 4096)|              0         |
|dropout (Dropout)|            (None, 4096)|              0         |
|dense (Dense)|                (None, 1)|                 4097      |

## Technologies
Project is created with
* Tensorflow 2.7

## Sources

The model was trained using dataset provided by Google: [Google Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset).

The model is based on the DCGAN as proposed in this [paper](https://arxiv.org/abs/1511.06434v1).
