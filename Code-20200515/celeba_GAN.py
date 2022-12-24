#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:49:24 2020

@author: scli
"""

import tensorflow as tf

import matplotlib.pyplot as plt
import os, time  
import numpy as np 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


# =============================================================================
# LOAD EXPLORE PREPARE DATA
# =============================================================================

# We will load 100'000 images from the celeb-a dataset and chose size 64x64
dir_data      = '/Users/scli/Desktop/celeba-dataset/img_align_celeba/img_align_celeba/' # directory with the data
Ntrain        = 100000 
nm_imgs       = np.sort(os.listdir(dir_data))
nm_imgs_train = nm_imgs[:Ntrain]
img_shape     = (64, 64, 3) 

	
# Load and show an example image 
from PIL import Image
# load the image
image = Image.open('/Users/scli/Desktop/celeba-dataset/img_align_celeba/img_align_celeba/000001.jpg')
# summarize some details about the image
print(image.format)
print(image.mode)
print(image.size)
# show the image
image.show()


# Example of pixel normalization
from numpy import asarray
from PIL import Image
# load image
image = Image.open('/Users/scli/Desktop/celeba-dataset/img_align_celeba/img_align_celeba/000001.jpg')
pixels = asarray(image)

# confirm pixel range is 0-255
print('Data Type: %s' % pixels.dtype)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
plt.imshow(pixels)

# convert from integers to floats
pixels = pixels.astype('float32')
print('Data Type: %s' % pixels.dtype)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
plt.imshow(pixels)
# to properly plot set range [0,1]
pixels_float = pixels/255
print('Data Type: %s' % pixels_float.dtype)
print('Min: %.3f, Max: %.3f' % (pixels_float.min(), pixels_float.max()))
plt.imshow(pixels_float)

# normalize to the range -1 to 1
pixels = (pixels- 127.5)/127.5
# confirm the normalization
print('Data Type: %s' % pixels.dtype)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
# show the image
plt.imshow(((pixels*127.5)+127.5)/255)


# Prepare data
def get_npdata(nm_imgs_train):
    X_train = []
    for i, myid in enumerate(nm_imgs_train):
        #image = load_img('/Users/scli/Desktop/celeba-dataset/img_align_celeba/img_align_celeba/',target_size=img_shape[:2])
        image = load_img(dir_data + "/" + myid,target_size=img_shape[:2])
        image = (img_to_array(image) - 127.5) / 127.5 # Normalize the images to [-1, 1]
        X_train.append(image)
    X_train = np.array(X_train)
    return(X_train)

X_train = get_npdata(nm_imgs_train)
print("X_train.shape = {}".format(X_train.shape))

# Show example of training data
plt.imshow(X_train[0])
plt.imshow(((X_train[0]*127.5)+127.5)/255)

BUFFER_SIZE = 100000
BATCH_SIZE = 128

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# =============================================================================
# Create the models
# =============================================================================

from tensorflow.keras import layers

# Generator

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(16*16*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 16, 256)))
    assert model.output_shape == (None, 16, 16, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)

    return model

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0])

# The Discriminator

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)


# Define the loss and optimizers

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# The discriminator and the generator optimizers are different since we will train two networks separately.
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Save checkpoints
# Checkpoints capture the exact value of all parameters (tf.Variable objects) used by a model.

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# =============================================================================
# Training
# =============================================================================



from IPython import display

EPOCHS = 15
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])
print(tf.shape(seed))


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)
  fig = plt.figure(figsize=(20,20))
  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow((predictions[i, :, :, :] * 127.5 + 127.5)/255)
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()


# To generate most recent images

seed_2 = tf.random.normal([num_examples_to_generate, noise_dim]) # you may choose a new seed

predictions = generator(seed_2, training=False)
fig = plt.figure(figsize=(20,20))
for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow((predictions[i, :, :, :]*127.5+127.5)/255)
    plt.axis('off')
      

# =============================================================================
# TRAINING
# =============================================================================

train(train_dataset, EPOCHS)

# Restore latest checkpoint, i.e. load the parameters from las tcheckpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# =============================================================================
# DISPLAY
# =============================================================================

import imageio
import PIL
import glob

# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

# display_image(12)

anim_file = 'celeb_gan_2.gif'

with imageio.get_writer(anim_file, mode='I', fps=2) as writer: #fps: frames per second
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = i #2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)








