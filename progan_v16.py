import datetime as dt
import os

import numpy as np

# Operations used in building the network. Many are not used in the current model
from ops import *
# FeedDict object used to continuously provide new training data
from feed_dict import FeedDict


# TODO: add argparser and flags


class ProGAN:
    def __init__(self,
            logdir,                    # directory of stored models
            imgdir,                    # directory of images for FeedDict
            learning_rate=0.001,       # Adam optimizer learning rate
            beta1=0,                   # Adam optimizer beta1
            beta2=0.99,                # Adam optimizer beta2
            w_lambda=10.0,             # WGAN-GP/LP lambda
            w_gamma=1.0,               # WGAN-GP/LP gamma
            epsilon=0.001,             # WGAN-GP/LP lambda
            z_length=512,              # latent variable size
            n_imgs=800000,             # number of images to show in each growth step
            batch_repeats=1,           # number of times to repeat minibatch
            n_examples=24,             # number of example images to generate
            lipschitz_penalty=True,    # if True, use WGAN-LP instead of WGAN-GP
            big_image=True,            # Generate a single large preview image, only works if n_examples = 24
            reset_optimizer=True,      # reset optimizer variables with each new layer
            batch_sizes=None,
            channels=None,
    ):

        # Scale down the number of factors if scaling_factor is provided
        self.channels = channels if channels else [512, 512, 512, 512, 256, 128, 64, 32, 16, 16]
        self.batch_sizes = batch_sizes if batch_sizes else [16, 16, 16, 16, 16, 16, 12, 4, 3]

        self.z_length = z_length
        self.n_examples = n_examples
        self.batch_repeats = batch_repeats if batch_repeats else 1
        self.n_imgs = n_imgs
        self.logdir = logdir
        self.big_image = big_image
        self.w_lambda = w_lambda
        self.w_gamma = w_gamma
        self.epsilon = epsilon
        self.reset_optimizer=reset_optimizer
        self.lipschitz_penalty = lipschitz_penalty

        # Initialize FeedDict
        self.feed = FeedDict.load(logdir, imgdir=imgdir, z_length=z_length, n_examples=n_examples)
        self.n_layers = self.feed.n_sizes
        self.max_imgs = (self.n_layers - 0.5) * self.n_imgs * 2

        # Initialize placeholders
        self.x_placeholder = tf.placeholder(tf.uint8, [None, 3, None, None])
        self.z_placeholder = tf.placeholder(tf.float32, [None, self.z_length])

        # Global step
        with tf.variable_scope('global_step'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)

        # Non-trainable variables for counting to next layer and incrementing value of alpha
        with tf.variable_scope('image_count'):
            self.total_imgs = tf.Variable(0, name='total_images', trainable=False, dtype=tf.int32)

            img_offset = tf.add(self.total_imgs, self.n_imgs)
            imgs_per_layer = self.n_imgs * 2

            self.img_step = tf.mod(img_offset, imgs_per_layer)
            self.layer = tf.minimum(tf.floor_div(img_offset, imgs_per_layer), self.n_layers - 1)

            fade_in = tf.to_float(self.img_step) / float(self.n_imgs)
            self.alpha = tf.minimum(1.0, tf.maximum(0.0, fade_in))

        # Initialize optimizer as member variable if not rest_optimizer, otherwise generate new
        # optimizer for each layer
        if self.reset_optimizer:
            self.lr = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
        else:
            self.g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2)
            self.d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2)
        self.networks = [self.create_network(i + 1) for i in range(self.n_layers)]

        # Initialize Session, FileWriter and Saver
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(self.logdir, graph=self.sess.graph)
        self.saver = tf.train.Saver()

        # Look in logdir to see if a saved model already exists. If so, load it
        try:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.logdir))
            print('Restored model -----------\n')
        except Exception:
            pass


    # Function for fading input of current layer into previous layer based on current value of alpha
    def reparameterize(self, x0, x1):
        return tf.add(
            tf.scalar_mul(tf.subtract(1.0, self.alpha), x0),
            tf.scalar_mul(self.alpha, x1)
        )


    # Build a generator for n layers
    def generator(self, z, n_layers):
        with tf.variable_scope('Generator'):

            with tf.variable_scope('latent_vector'):
                z = tf.expand_dims(z, 2)
                g1 = tf.expand_dims(z, 3)

            for i in range(n_layers):
                with tf.variable_scope('layer_{}'.format(i)):

                    if i == n_layers - 1:
                        g0 = g1

                    with tf.variable_scope('1'):
                        if i == 0:
                            g1 = g_conv_layer(g1, self.channels[i],
                                              filter_size=4, padding='VALID', mode='transpose',
                                              output_shape=[tf.shape(g1)[0], self.channels[i], 4, 4])
                        else:
                            g1 = g_conv_layer(g1, self.channels[i], mode='upscale')

                    with tf.variable_scope('2'):
                        g1 = g_conv_layer(g1, self.channels[i])

            with tf.variable_scope('rgb_layer_{}'.format(n_layers - 1)):
                g1 = conv(g1, 3, filter_size=1)

            if n_layers > 1:
                with tf.variable_scope('rgb_layer_{}'.format(n_layers - 2)):
                    g0 = conv(g0, 3, filter_size=1)
                    g0 = upscale(g0)
                    g = self.reparameterize(g0, g1)
            else:
                g = g1

        return g


        # Build a discriminator n layers
    def discriminator(self, x, n_layers):
        with tf.variable_scope('Discriminator'):

            if n_layers > 1:
                with tf.variable_scope('rgb_layer_{}'.format(n_layers - 2)):
                    d0 = downscale(x)
                    d0 = d_conv_layer(d0, self.channels[n_layers - 1], filter_size=1)

            with tf.variable_scope('rgb_layer_{}'.format(n_layers - 1)):
                d1 = d_conv_layer(x, self.channels[n_layers], filter_size=1)

            for i in reversed(range(n_layers)):
                with tf.variable_scope('layer_{}'.format(i)):

                    if i == 0:
                        d1 = minibatch_stddev(d1)

                    with tf.variable_scope('1'):
                        d1 = d_conv_layer(d1, self.channels[i])

                    with tf.variable_scope('2'):
                        if i == 0:
                            d1 = d_conv_layer(d1, self.channels[0],
                                              filter_size=4, padding='VALID')
                        else:
                            d1 = d_conv_layer(d1, self.channels[i], mode='downscale')

                    if i == n_layers - 1 and n_layers > 1:
                        d1 = self.reparameterize(d0, d1)

            with tf.variable_scope('dense'):
                d = tf.reshape(d1, [-1, self.channels[0]])
                d = dense(d, 1)

        return d


    # Function for creating network layout at each layer
    def create_network(self, n_layers):

        # image dimensions
        dim = 2 ** (n_layers + 1)

        # Build the current network
        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            Gz = self.generator(self.z_placeholder, n_layers)
            Dz = self.discriminator(Gz, n_layers)

            # Mix different resolutions of input images according to value of alpha
            with tf.variable_scope('training_images'):
                x = scale_uint8(self.x_placeholder)
                if n_layers > 1:
                    x0 = upscale(downscale(x))
                    x1 = x
                    x = self.reparameterize(x0, x1)

            Dx = self.discriminator(x, n_layers)

            # Fake and real image mixing for WGAN-GP loss function
            interp = tf.random_uniform(shape=[tf.shape(Dz)[0], 1, 1, 1], minval=0.0, maxval=1.0)
            x_hat = interp * x + (1 - interp) * Gz
            Dx_hat = self.discriminator(x_hat, n_layers)

        # Loss function and scalar summaries
        with tf.variable_scope('Loss_Function'):

            # Wasserstein Distance
            wd = Dz - Dx

            # Gradient/Lipschitz Penalty
            grads = tf.gradients(Dx_hat, [x_hat])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), [1, 2, 3]))

            if self.lipschitz_penalty:
                gp = tf.square(tf.maximum((slopes - self.w_gamma) / self.w_gamma, 0))
            else:
                gp = tf.square((slopes - self.w_gamma) / self.w_gamma)

            gp_scaled = self.w_lambda * gp

            # Epsilon penalty keeps discriminator output for drifting too far away from zero
            epsilon_cost = self.epsilon * tf.square(Dx)

            # Cost and summary scalars
            g_cost = tf.reduce_mean(-Dz)
            d_cost = tf.reduce_mean(wd + gp_scaled + epsilon_cost)
            wd = tf.abs(tf.reduce_mean(wd))
            gp = tf.reduce_mean(gp)

            # Summaries
            wd_sum = tf.summary.scalar('Wasserstein_distance_{}_({}x{})'.format(
                n_layers - 1, dim, dim), wd)
            gp_sum = tf.summary.scalar('gradient_penalty_{}_({}x{})'.format(
                n_layers - 1, dim, dim), gp)

        # Collecting variables to be trained by optimizers
        g_vars, d_vars = [], []
        var_scopes = ['layer_{}'.format(i) for i in range(n_layers)]
        var_scopes.extend([
            'dense',
             'rgb_layer_{}'.format(n_layers - 2),
             'rgb_layer_{}'.format(n_layers - 1)
        ])

        for scope in var_scopes:
            g_vars.extend(tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='Network/Generator/{}'.format(scope)
            ))
            d_vars.extend(tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='Network/Discriminator/{}'.format(scope)
            ))

        # Generate optimizer operations
        # if self.reset_optimizer is True then initialize a new optimizer for each layer
        with tf.variable_scope('Optimize'):
            if self.reset_optimizer:
                g_train = tf.train.AdamOptimizer(
                    self.lr, self.beta1, self.beta2, name='G_optimizer_{}'.format(n_layers - 1)
                ).minimize(
                    g_cost, var_list=g_vars)
                d_train = tf.train.AdamOptimizer(
                    self.lr, self.beta1, self.beta2, name='D_optimizer_{}'.format(n_layers - 1)
                ).minimize(
                    d_cost, var_list=d_vars, global_step=self.global_step)

            else:
                g_train = self.g_optimizer.minimize(g_cost, var_list=g_vars)
                d_train = self.d_optimizer.minimize(d_cost, var_list=d_vars, global_step=self.global_step)

            # Increment image count
            n_imgs = tf.shape(x)[0]
            new_image_count = tf.add(self.total_imgs, n_imgs)
            img_step_op = tf.assign(self.total_imgs, new_image_count)
            d_train = tf.group(d_train, img_step_op)

        # Print variable names to before running model
        print('\nGenerator variables for layer {} ({} x {}):'.format(n_layers - 1, dim, dim))
        print([var.name for var in g_vars])
        print('\nDiscriminator variables for layer {} ({} x {}):'.format(n_layers - 1, dim, dim))
        print([var.name for var in d_vars])

        # Generate preview images
        with tf.variable_scope('image_preview'):
            n_real_imgs = min(self.batch_sizes[n_layers - 1], 4)
            fake_imgs = tensor_to_imgs(Gz)
            real_imgs = tensor_to_imgs(x[:n_real_imgs])

            # Upsize images to normal visibility
            if dim < 256:
                fake_imgs = resize_images(fake_imgs, (256, 256))
                real_imgs = resize_images(real_imgs, (256, 256))

            # Concatenate images into one large image for preview, only used if 24 preview images are requested
            if self.big_image and self.n_examples == 24:
                fake_img_list = tf.unstack(fake_imgs, num=24)
                fake_img_list = [tf.concat(fake_img_list[6 * i:6 * (i + 1)], 1) for i in range(4)]
                fake_imgs = tf.concat(fake_img_list, 0)
                fake_imgs = tf.expand_dims(fake_imgs, 0)

                real_img_list = tf.unstack(real_imgs, num=n_real_imgs)
                real_imgs = tf.concat(real_img_list, 1)
                real_imgs = tf.expand_dims(real_imgs, 0)

            # images summaries
            fake_img_sum = tf.summary.image('fake{}x{}'.format(dim, dim), fake_imgs, self.n_examples)
            real_img_sum = tf.summary.image('real{}x{}'.format(dim, dim), real_imgs, 4)

        return dict(
            wd=wd, gp=gp, wd_sum=wd_sum, gp_sum=gp_sum, g_train=g_train, d_train=d_train,
            fake_img_sum=fake_img_sum, real_img_sum=real_img_sum, Gz=Gz
        )


    # Get current layer, global step, alpha and total number of images used so far
    def get_global_vars(self):
        gs, layer, img_step, alpha, total_imgs = self.sess.run([
            self.global_step, self.layer, self.img_step, self.alpha, self.total_imgs
        ])
        if layer == 0: img_step -= self.n_imgs
        return gs, layer, img_step, alpha, total_imgs


    def get_layer_ops(self, layer):
        dim = 2 ** (layer + 2)
        batch_size = self.batch_sizes[layer]
        n_imgs = self.n_imgs
        if layer > 0: n_imgs *= 2

        layer_ops = self.networks[layer]
        g_train = layer_ops.get('g_train')
        d_train = layer_ops.get('d_train')
        get_ops = lambda *op_names: [layer_ops.get(name) for name in op_names]
        scalar_sum_ops = get_ops('wd', 'gp', 'wd_sum', 'gp_sum')
        img_sum_ops = get_ops('fake_img_sum', 'real_img_sum')

        return dim, batch_size, n_imgs, g_train, d_train, scalar_sum_ops, img_sum_ops


    # Main training function
    def train(self, save_interval=80000):

        def get_loop_progress(layer, img_step):
            percent_done = img_step / self.n_imgs
            if layer > 0: percent_done /= 2
            time = dt.datetime.now()
            return time, percent_done

        gs, prev_layer, img_step, alpha, total_imgs = self.get_global_vars()
        start_time, start_percent_done = get_loop_progress(prev_layer, img_step)
        dim, batch_size, n_imgs, g_train, d_train, scalar_sum_ops, img_sum_ops = self.get_layer_ops(prev_layer)

        save_step = (total_imgs // save_interval + 1) * save_interval

        while total_imgs < self.max_imgs:
            gs, layer, img_step, alpha, total_imgs = self.get_global_vars()

            # Get network operations and loss functions for current layer
            if layer != prev_layer:
                start_time, start_percent_done = get_loop_progress(prev_layer, img_step)
                dim, batch_size, n_imgs, g_train, d_train, scalar_sum_ops, img_sum_ops = self.get_layer_ops(layer)

            # Get training data and latent variables to store in feed_dict
            feed_dict = {
                self.x_placeholder: self.feed.x_batch(batch_size, dim),
                self.z_placeholder: self.feed.z_batch(batch_size)
            }

            # Here's where we actually train the model
            for _ in range(self.batch_repeats):
                self.sess.run(d_train, feed_dict)
                self.sess.run(g_train, feed_dict)

            if gs % 20 == 0:

                # Get loss values and summaries
                wd_value, gp_value, wd_sum_str, gp_sum_str = self.sess.run(scalar_sum_ops, feed_dict)

                # Print current status, loss functions, etc.
                time, percent_done = get_loop_progress(layer, img_step)
                print(
                    'dimensions: ({} x {}) ---- {}% ---- images: {}/{} ---- alpha: {} ---- global step: {}'
                    '\nWasserstein distance: {}\ngradient penalty: {}'.format(
                        dim, dim, np.round(percent_done * 100, 4), img_step, n_imgs,
                        np.round(alpha, 4), gs, wd_value, gp_value
                ))

                # Calculate and print estimated time remaining
                delta_t = time - start_time
                time_remaining = delta_t * (1 / (percent_done - start_percent_done + 1e-8) - 1)
                print('est. time remaining on layer {}: {}\n'.format(layer, time_remaining))

                # Log scalar data every 20 global steps
                self.writer.add_summary(wd_sum_str, gs)
                self.writer.add_summary(gp_sum_str, gs)

            # Operations to run every save interval
            if total_imgs > save_step:
                save_step += save_interval

                # Save the model and generate image previews
                print('\nsaving and making images...\n')
                self.saver.save(
                    self.sess, os.path.join(self.logdir, "model.ckpt"),
                    global_step=self.global_step
                )
                self.feed.save()

                img_preview_feed_dict = {
                    self.x_placeholder: feed_dict[self.x_placeholder][:4],
                    self.z_placeholder: self.feed.z_fixed
                }

                fake_img_sum_str, real_img_sum_str = self.sess.run(
                    img_sum_ops, img_preview_feed_dict
                )
                self.writer.add_summary(fake_img_sum_str, gs)
                self.writer.add_summary(real_img_sum_str, gs)

            prev_layer = layer


    def get_cur_res(self):
        cur_layer = self.sess.run(self.layer)
        return 2 ** (2 + cur_layer)


    def generate(self, z):
        solo = z.ndim == 1
        if solo:
            z = np.expand_dims(z, 0)

        cur_layer = int(self.sess.run(self.layer))
        imgs = self.networks[cur_layer][9]
        imgs = self.sess.run(imgs, {self.z_placeholder: z})

        if solo:
            imgs = np.squeeze(imgs, 0)
        return imgs


if __name__ == '__main__':
    # progan = ProGAN(logdir='logdir_v5', imgdir='memmaps')

    # progan = ProGAN(logdir='logdir_v6', imgdir='memmaps', batch_repeats=4)

    progan = ProGAN(logdir='logdir_v8', imgdir='memmaps', batch_repeats=4)
    # progan = ProGAN(logdir='logdir_v9', imgdir='memmaps', batch_repeats=4, batch_sizes=[128, 128, 128, 64, 32, 16, 12, 8, 4])

    progan.train()