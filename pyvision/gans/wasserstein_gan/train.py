#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import importlib
wgan = importlib.import_module("wgan")
from wgan import WGAN, Generator, Discriminator, batch_size, dataroot
import torch
import os
import torch.nn as nn
import torch.autograd
import torch.optim as optim
import torchvision.utils
import numpy as np
import pickle
import glob, os, sys
import datetime
import matplotlib
# %matplotlib inline
matplotlib.use('agg')
import matplotlib.pyplot as plt
import subprocess as sp
import argparse

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# In[2]:


#get_ipython().system('ls')


# Main Class

# In[3]:


class CelebA(object):
    def __init__(self, train_params, ckpt_params, gan_params):
        # Training parameters
        self.root_dir = train_params['root_dir']
        self.gen_dir = train_params['gen_dir']
        self.batch_size = train_params['batch_size']
        self.train_len = train_params['train_len']
        self.learning_rate = train_params['learning_rate']
        self.momentum = train_params['momentum']
        #self.use_cuda = train_params['use_cuda']
        self.batch_report_interval = ckpt_params['batch_report_interval']
        self.ckpt_path = ckpt_params['ckpt_path']
        self.save_stats_interval = ckpt_params['save_stats_interval']
        if not os.path.isdir(self.ckpt_path):
            os.mkdir(self.ckpt_path)
        if not os.path.isdir(self.gen_dir):
            os.mkdir(self.gen_dir)
        self.latent_dim = gan_params['latent_dim']
        self.n_critic = gan_params['n_critic']
        self.num_batches = self.train_len // self.batch_size
        self.compile()



    def compile(self):
      self.gan = WGAN(self.latent_dim, self.batch_size)
      self.G_optimizer = optim.RMSprop(self.gan.G.parameters(),
                lr=self.learning_rate)
      self.D_optimizer = optim.RMSprop(self.gan.D.parameters(),
                lr=self.learning_rate)


      #if torch.cuda.is_available():
      self.gan = self.gan.to(device)
      self.latent_vars = []
      for i in range(100):
              self.latent_vars.append(self.gan.create_latent_var(1))
    def save_stats(self, stats):
        """Save model statistics"""

        fname_pkl = '{}/{}-stats.pkl'.format(self.ckpt_path, "WGAN")
        print('Saving model statistics to: {}'.format(fname_pkl))
        with open(fname_pkl, 'wb') as fp:
            pickle.dump(stats, fp)
    def eval(self, n, epoch=None, while_training=False):
      self.gan.G.eval()
      m = int(np.sqrt(n))
      for i in range(n):
            # Reuse fixed latent variables to keep random process intact
            if while_training:
                img = self.gan.generate_img(self.latent_vars[i])
            else:
                img = self.gan.generate_img()
            img = wgan.unnormalize(img.squeeze())
            fname_in = '{}/test{:d}.png'.format(self.ckpt_path, i)
            torchvision.utils.save_image(img, fname_in)
      stack = 'montage {}/test* -tile {}x{} -geometry 64x64+1+1             #{}/epoch'.format(self.ckpt_path, m, m, self.ckpt_path)
      stack = stack + str(epoch + 1) + '.png' if epoch is not None else stack + '.png'

      #sp.call(stack.split())
      #for f in glob.glob('{}/test*'.format(self.ckpt_path)):
            #os.remove(f)

    def train(self, nb_epochs, data_loader):
        """Train model on data"""

        # Initialize tracked quantities and prepare everything
        G_all_losses, D_all_losses, times = [], [], wgan.AvgMeter()
        wgan.format_hdr(self.gan, self.root_dir, self.train_len)
        start = datetime.datetime.now()
        g_iter, d_iter = 0, 0

        for epoch in range(nb_epochs):
            logs = {}
            print('EPOCH {:d} / {:d}'.format(epoch + 1, nb_epochs))
            G_losses, D_losses = wgan.AvgMeter(), wgan.AvgMeter()
            start_epoch = datetime.datetime.now()

            avg_time_per_batch = wgan.AvgMeter()
            # Mini-batch SGD
            for batch_idx, (x, _) in enumerate(data_loader):

                # Critic update ratio

                n_critic = 20 if g_iter < 50 or (g_iter + 1) % 500 == 0 else self.n_critic

                self.gan.G.train()

                batch_start = datetime.datetime.now()

                wgan.progress_bar(batch_idx, self.batch_report_interval,
                    G_losses.avg, D_losses.avg)

                #if torch.cuda.is_available():
                x = x.to(device)

                # Update discriminator
                D_loss, fake_imgs = self.gan.train_D(x, self.D_optimizer, self.batch_size)
                D_losses.update(D_loss, self.batch_size)
                d_iter += 1

                # Update generator
                if batch_idx % n_critic == 0:
                    G_loss = self.gan.train_G(self.G_optimizer, self.batch_size)
                    G_losses.update(G_loss, self.batch_size)
                    g_iter += 1

                batch_end = datetime.datetime.now()
                batch_time = int((batch_end - batch_start).total_seconds() * 1000)
                avg_time_per_batch.update(batch_time)

                if (batch_idx % self.batch_report_interval == 0 and batch_idx) or                     self.batch_report_interval == self.num_batches:
                    G_all_losses.append(G_losses.avg)
                    D_all_losses.append(D_losses.avg)
                    wgan.show_learning_stats(batch_idx, self.num_batches, G_losses.avg, D_losses.avg, avg_time_per_batch.avg)
                    [k.reset() for k in [G_losses, D_losses, avg_time_per_batch]]
                    self.eval(100, epoch=epoch, while_training=True)
                    # print('Critic iter: {}'.format(g_iter))


                if batch_idx % self.save_stats_interval == 0 and batch_idx:
                    stats = dict(G_loss=G_all_losses, D_loss=D_all_losses)
                    self.save_stats(stats)
            wgan.clear_line()
            print('Elapsed time for epoch: {}'.format(wgan.time_elapsed_since(start_epoch)))
            self.gan.save_model(self.ckpt_path, epoch)
            self.eval(100, epoch=epoch, while_training=True)

        # Print elapsed time
        elapsed = wgan.time_elapsed_since(start)
        print('Training done! Total elapsed time: {}\n'.format(elapsed))




# In[4]:


gan_params = {
        'latent_dim': 100, #size of latent dimension, 100 is to achieve best result for this Architecture
        'n_critic': 5  # Discriminator update per Generator update
    }
train_params = {
        'root_dir': dataroot,
        'gen_dir': 'gen',
        'batch_size': batch_size,
        'train_len': 96000,  # Training length on the dataset
        'learning_rate': 0.00001,  # Learning rate
        'momentum': (0.5, 0.999),
    }
ckpt_params = {
        'batch_report_interval': 100,
        'ckpt_path': 'ckpt',
        'save_stats_interval': 500
    }


# In[5]:


#model = CelebA(train_params, ckpt_params, gan_params)
#data_loader = wgan.load_dataset()

torch.manual_seed(100)
#n_epoch = 135 #Number of epochs to train for
#model.train(n_epoch, data_loader)


# In[ ]:
