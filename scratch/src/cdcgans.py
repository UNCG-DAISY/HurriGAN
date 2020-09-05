# authors: Andres Pitta, Braden Tam, Florence Wang and Hanying Zhang

# General
import numpy as np
import os
import time
import random

# For modeling the Neural Network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Reshape, Conv2DTranspose, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

# For images 
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.image import imread
from IPython import display

# For the labels
from sklearn.preprocessing import LabelEncoder


class cdcgans():
  """
  Conditional DCGANS.
  """
  def __init__(self, img_size, noise_dim = 100, n_classes = 10,
               lr_g = 0.0002, lr_d = 0.0002, beta1 = 0.5, num_examples_to_generate = 10,
               epochs = 50, batch_size = 32, dropout_rate = 0.3, restore_checkpoint = False,
               save_ckpt_path = './training_checkpoints', load_ckp_path = './training_checkpoints',
               city_names = ['Boston']):
    '''
    Init function.
    Parameters
    --------------
    img_size (int):
        Input image size (width and height)
    noise_dim (int):
        Input noise dimension to generate fake image (latent dimension).
        Default 100.
    n_classes (int):
        Total number of classes (label) in the dataset.
        Default 10.
    lr_g (float):
        Adam optimizer learning rate for generator network.
        Default 0.0002.
    lr_d (float):
        Adam optimizer learning rate for discriminator network.
        Default 0.0002.
    beta1 (float):
        The exponential decay rate for the 1st moment estimates in Adam optimizer.
        Default 0.5.
    num_examples_to_generate (int):
        Number of images to generate along with training.
        Default 10.
    epochs (int):
        Number of epochs.
        Default 50.
    batch_size (int):
        Batch size used in the dataset.
        Dafault 32.
    dropout_rate (float):
        Percentage to dropout in the discriminator network.
        Default 0.3.
    restore_checkpoint (bool):
        Whether to restore checkpoints.
        Default False.
    save_ckpt_path (str):
        Folder path to save checkpoints.
        Default ./training_checkpoints.
    load_ckp_path (str):
        Folder path to restore checkpoints.
        Default ./training_checkpoints.
    city_names (list):
        Name of the conditioning label
        Default ['Boston']
    
    '''
    # data input
    self.img_size = img_size
    self.channels = 3
    self.noise_dim = noise_dim
    self.n_classes = n_classes
    self.city_names = city_names

    # epochs and batches
    self.epochs = epochs
    self.batch_size = batch_size

    # build generator and discriminator
    self.generator = self.make_generator_model()
    self.discriminator = self.make_discriminator_model(dropout_rate)
    
    # set optimizer   
    self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate = lr_g, beta_1 = beta1)
    self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = lr_d, beta_1 = beta1)

    # We will reuse this seed overtime
    # to visualize progress in the animated GIF)
    self.seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # checkpoints 
    # save checkpoint
    checkpoint_dir = save_ckpt_path
    self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                discriminator_optimizer=self.discriminator_optimizer,
                                generator=self.generator,
                                discriminator=self.discriminator)
    # load checkpoint
    if restore_checkpoint:
      self.checkpoint.restore(tf.train.latest_checkpoint(load_ckp_path))
      print('Successfully restored checkpoint!')

  #generator model
  def make_generator_model(self):
    '''
    Build tensorflow architecture of generator model.
    '''
    # note: layers shape depends on image size
    model = tf.keras.Sequential()
    model.add(layers.Dense(self.img_size // 4 * self.img_size // 4 * 512, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((self.img_size // 4, self.img_size // 4, 512)))
    assert model.output_shape == (None, self.img_size // 4, self.img_size // 4, 512)

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, self.img_size // 4, self.img_size // 4, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, self.img_size // 2, self.img_size // 2, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, self.img_size, self.img_size, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, self.img_size, self.img_size, 3)
    #model.summary()

    noise=layers.Input(shape=(100,))
    
    label=layers.Input(shape=(1,))
    label_embedding=layers.Flatten()(layers.Embedding(self.n_classes, 100)(label))
    
    model_input=layers.multiply([noise, label_embedding])

    img=model(model_input)
    generator_model = Model([noise,label],img)

    #generator_model.summary()
    return generator_model

  # discriminator model
  def make_discriminator_model(self, dropout_rate):
    '''
    Build tensorflow architecture of discriminator model.
    Parameters
    --------------
    dropout_rate (float):
        Percentage to dropout.
    '''
    in_label = layers.Input(shape=(1,))
    in_label = layers.Input(shape=(1,))
    in_image = layers.Input(shape=(self.img_size, self.img_size, self.channels))

    model = GaussianNoise(0.2)(in_image)
    model = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(model)
    model = layers.LeakyReLU()(model)
    model = layers.Dropout(dropout_rate)(model)
    
    model = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(model)
    model = layers.LeakyReLU()(model)
    model = layers.Dropout(dropout_rate)(model)

    model = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')(model)
    model = layers.LeakyReLU()(model)
    model = layers.Dropout(dropout_rate)(model)

    model = layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same')(model)
    model = layers.LeakyReLU()(model)
    model = layers.Dropout(dropout_rate)(model)

    model = Flatten()(model)
    model = Dense(1000)(model)
    model = layers.LeakyReLU()(model)
    model = layers.Dropout(dropout_rate)(model)
    li = layers.Embedding(self.n_classes, 1000)(in_label)
    li = layers.Flatten()(li)

    merge = layers.Multiply()([model, li])

    model = layers.Dense(512)(merge)
    model = layers.LeakyReLU(0.2)(model)
    model = layers.Dropout(0.3)(model)
    out = layers.Dense(1)(model)

    model = tf.keras.Model(inputs=[in_image,in_label], outputs=out)
   # model.summary()
    return model

    # discriminator loss function
  def discriminator_loss(self, real_output, fake_output):
    '''
    Calculates binary crossentropy loss for discriminator.
    Parameters
    -------------------
    real_output (tensorflow numpy array):
        Discriminator output from fitting real images.
    fake_output (tensorflow numpy array):
        Discriminator output from fitting fake images.
    Returns
    ---------------
        Average binary crossentropy loss of discriminator.
    '''
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    
    return tf.reduce_mean(0.5*total_loss)
    
  # generator loss function
  def generator_loss(self, fake_output):
    '''
    Calculates binary crossentropy loss for generator.
    Parameters
    -------------------
    fake_output (tensorflow numpy array):
        Discriminator output from fitting fake images.
    Returns
    ---------------
        Average binary crossentropy loss of generator.   
    '''
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    return tf.reduce_mean(cross_entropy(tf.ones_like(fake_output), fake_output))  
  
  # Notice the use of `tf.function`
  # This annotation causes the function to be "compiled".
  @tf.function
  def train_step(self, images, labels):
    '''
    Compile generator and discriminator.
    Parameters
    -------------------
    images (np.array):
        Training images.
    labels (np.array):
        Labels of images.
    Returns
    ---------------
    gen_loss (float):
        Average binary crossentropy loss of generator.
    disc_loss (float):
        Average binary crossentropy loss of generator.
    fake_output (tensorflow numpy array):
        Discriminator output from fitting fake images.
    '''
    noise = tf.random.normal([self.batch_size, self.noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = self.generator([noise,labels], training=True)

      real_output = self.discriminator([images,labels], training=True)
      fake_output = self.discriminator([generated_images,labels], training=True)

      gen_loss = self.generator_loss(fake_output)
      disc_loss = self.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

    self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
    self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    return gen_loss, disc_loss, fake_output
    
  # train the model
  def train(self, dataset, steps_per_epoch, labels,
            saving_gap = 15,
            save_img = True, img_path = './generated_imgs', 
            save_loss_batch = False, save_loss_epoch = True, loss_path = './losses'):
    '''
    Train the model and print results over each epochs.
    Parameters
    ------------------- 
    dataset (np.array, np.array):
        Image generator tuple with the images and its labels (images, labels).  
    steps_per_epoch (int):
        Number of steps per epoch (total images / batch size).
    labels (np.array):
        Input labels for sample images to generate.
    saving_gap (int):
        Gaps to save checkpoints, images and losses over epochs. Default 15.
    save_img (bool):
        Whether to save images produced by 'generate_and_save_images' function 
        for each 'saving_gap' epochs. Default True.
    img_path (str):
        Folder path to save images. Default './generated_imgs'.
    save_loss_batch (bool):
        Whether to save plots of losses over batches produced by 'summarize_epoch' function 
        for each 'saving_gap' epochs. Default False.
    save_loss_epoch (bool):
        Whether to save the plot of total loss over epochs produced by 'plot_loss_over_epoch' function.
        Default True.
    loss_path (str):
        Folder path to save losses. Default './losses'.
    '''

    # save losses over epochs
    losses_on_epoch = {'g_losses' :[], 'd_losses': [], 'epochs': [],'perc_fake_true':[]}

    for epoch in range(self.epochs):
      print(f"Starting epoch {epoch+1}.")
      start = time.time()
      #save losses over batches
      g_losses = []
      d_losses = []
      num_fake_true = [] # number of classify fake image as real

      batch = 0

      for image_batch, image_label in dataset: 
        if batch % 50 == 0:
          print(f"  Starting batch {batch+1}/{steps_per_epoch}")
        # training part
        g_loss, d_loss, fake_arr = self.train_step(image_batch,image_label)
        # get number of images that fools discriminator
        fake_true = np.sum(fake_arr.numpy() > 0) 
        # append losses over batches
        g_losses.append(g_loss.numpy())
        d_losses.append(d_loss.numpy())
        num_fake_true.append(fake_true)##

        batch += 1
        if batch >= steps_per_epoch:
          break
      print("  Done train step.")

      # Produce images for the GIF as we go
      display.clear_output(wait=True)
      # save images for each epoch
      self.generate_and_save_images(self.generator,
                              epoch + 1,
                              self.seed,
                              labels,
                              img_path,
                              save_img,
                              1)
      
      # Save the model every 15 epochs
      if saving_gap != 0:
        if ((epoch + 1) % saving_gap == 0) | ((epoch + 1) == self.epochs):
          self.checkpoint.save(file_prefix = self.checkpoint_prefix)
      # plot loss on each epoch
      g_loss_mean, d_loss_mean, perc_fake_true = self.summarize_epoch(epoch, 
                                                                      d_losses, g_losses, 
                                                                      steps_per_epoch,
                                                                      num_fake_true,
                                                                      loss_path,
                                                                      save_loss_batch,
                                                                      saving_gap)
      # save the losses
      losses_on_epoch['g_losses'].append(g_loss_mean)
      losses_on_epoch['d_losses'].append(d_loss_mean)
      losses_on_epoch['epochs'].append(epoch + 1)
      losses_on_epoch['perc_fake_true'].append(perc_fake_true)
      
      print ('Time taken for epoch {} is {} sec.'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    self.generate_and_save_images(self.generator,
                            self.epochs,
                            self.seed,
                            labels,
                            img_path,
                            True, # default save last image
                            saving_gap)
    # plot loss after all epochs
    self.plot_loss_over_epoch(losses_on_epoch['g_losses'],losses_on_epoch['d_losses'],losses_on_epoch['epochs'], 
                        losses_on_epoch['perc_fake_true'],
                        loss_path,save_loss_epoch)
    # print loss after all epochs
    print('g_loss: {} \nd_loss: {} \npercentage of classify fake image as real: {}'.format(
        losses_on_epoch['g_losses'][-1],
        losses_on_epoch['d_losses'][-1],
        losses_on_epoch['perc_fake_true'][-1]))
    
  #produce a new image
  def produce_image(self,label):
    '''
    Generate a new fake image and plot the image.
    Parameters
    ------------------- 
    label (np.array):
        The label for the image to generate (1-D numpy array with lenghth 1).
    
    Returns
    ------------------- 
    img_array (np.array):
        Numpy array of the generated image.
    '''
    generator = self.generator
    noise = tf.random.normal([1, self.noise_dim])
    generated_image = generator([noise,np.array([label])], training=False)
    img_array = generated_image[0, :, :].numpy()
    img_array = (img_array - np.min(img_array))/np.ptp(img_array)
    plt.imshow(img_array)
    return img_array

  ### below are functions used in the train function

  # function used in train
  def generate_and_save_images(self, model, epoch, test_input, labels, 
                               img_path, save_img = True,saving_gap = 15):
    '''
    Generate, plot and save fake images with subplot 3x3 along with training. 
    Parameters
    ------------------- 
    model (tensorflow model):
        The generator used to generate images.
    epoch (int):
        Current epoch in training step.
    test_input (np.array): 
        The input random noise to generate image.
    labels (np.array):
        The input labels for sample images to generate.
    img_path (str):
        Folder path to save images.
    save_img (bool):
        Whether to save images for each 'saving_gap' epochs. Default True.
    saving_gap (int):
        Gaps to save checkpoints, images and losses over epochs. Default 15.
    '''
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    fig = plt.figure(figsize=(8,10))
    fig.suptitle(f'Generation at epoch {epoch}', fontsize=12)
    labels_enc = LabelEncoder().fit_transform(labels)

    for i in range(len(labels)):
        predictions = model([test_input,np.array([labels_enc[i]]).reshape((-1,1))], training=False)
        plt.subplot(5, 2, i+1) ## NOTE: This depends on num_examples_to_generate
        p = predictions[0, :, :]
        p = (p - np.min(p))/np.ptp(p)
        plt.imshow(p)
        plt.title('label: {}'.format(self.city_names[i]))
        plt.axis('off')
    if (save_img) & (saving_gap != 0):
      # save images on each 15 epochs
      if (epoch == 0 ) | ((epoch + 1) % saving_gap == 0) | ((epoch) == self.epochs):
          plt.savefig(img_path + '/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
    plt.close()

  # function used in train
  # plot loss after all epochs
  def plot_loss_over_epoch(self, g_losses,d_losses,epochs, fake_true,
                           loss_path, save_loss_epoch = True):
    '''
    Plot and save total loss over epochs after training. 
    Parameters
    ------------------- 
    g_losses (list):
        Generator losses over epochs.
    d_losses (list):
        Discriminator losses over epochs.
    fake_true (list):
        Percentage of fake images that fools discriminator over epochs.
    loss_path (str):
        Folder path to save losses.
    save_loss_epoch (bool):
        Whether to save the plot of total loss over epochs. Default True.
    '''

    fig, ax = plt.subplots()
    plt.plot(g_losses, label='Generator', alpha=0.6)
    plt.plot(d_losses, label='Discriminator', alpha=0.6)
    #plt.plot(fake_true, label = 'perc of fools', alpha=0.6)
    plt.title("Losses")
    plt.xticks(epochs)
    plt.legend()
    if save_loss_epoch:
      plt.savefig(loss_path +'/{:04d}_epochs_losses.png'.format(epochs[-1]))
    plt.show()
    plt.close()
  
  # function used in train
  # reference: https://github.com/gsurma/image_generator/blob/master/ImageGeneratorDCGAN.ipynb
  # plot loss on the each epoch
  def summarize_epoch(self, epoch, g_losses, d_losses, steps_per_epoch, fake_true,
                      loss_path, save_loss_batch = False, saving_gap = 15):
      '''
      Plot and save fake loss over bathces in each epoch along with training.
      Parameters
      ------------------- 
      epoch (int):
          Current epoch in training step.
      g_losses (list):
        Generator losses over batches on current epoch.
      d_losses (list):
        Discriminator losses over batches on current epoch.
      fake_true (list):
        Percentage of fake images that fools discriminator over batches on current epoch.
      loss_path (str):
        Folder path to save losses.
      save_loss_batch (bool):
        Whether to save plots of losses over batches for each 'saving_gap' epochs. Default False.
      saving_gap (int):
          Gaps to save checkpoints, images and losses over epochs. Default 15.
      '''
      g_loss_mean = np.mean(g_losses[-steps_per_epoch:])
      d_loss_mean = np.mean(d_losses[-steps_per_epoch:])
      perc_fake_true = np.sum(fake_true)/(steps_per_epoch * self.batch_size)
      print("Epoch {}/{}".format(epoch + 1, self.epochs),
            "\nG Loss: {:.5f}".format(g_loss_mean),
            "\nD Loss: {:.5f}".format(d_loss_mean),
            "\nPercent of fake true: {:.5f}".format(perc_fake_true))#
      fig, ax = plt.subplots()
      plt.plot(g_losses, label='Generator', alpha=0.6)
      plt.plot(d_losses, label='Discriminator', alpha=0.6)
      #percent of fake true over epochs
      perc_fake_true_batch = [i/self.batch_size for i in fake_true]
      plt.plot(perc_fake_true_batch, label = 'perc of fools',alpha = 0.6)

      plt.title("Losses")
      plt.legend()
      # save losses each 15 epochs
      if (save_loss_batch) & (saving_gap != 0) :
        if (epoch ==0 ) | ((epoch + 1) % saving_gap == 0) | (epoch == self.epochs):
          plt.savefig(loss_path + '/losses_at_epoch_{:04d}.png'.format(epoch+1))
      plt.show()
      plt.close()
      return g_loss_mean, d_loss_mean, perc_fake_true