import os
import datetime as dt

# Operations used in building the network. Many are not used in the current model
from ops import *
# FeedDict object used to continuously provide new training data
from feed_dict import FeedDict


# TODO: add argparser and flags
# TODO: refactor training function
# TODO: train next version of model using reset_optimizer=True


class ProGAN:
    def __init__(self,
            logdir,                    # directory of stored models
            img_dir,                   # directory of images for FeedDict
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
            lipschitz_penalty=True,   # if True, use WGAN-LP instead of WGAN-GP
            big_image=True,            # Generate a single large preview image, only works if n_examples = 24
            scaling_factor=None,       # factor to scale down number of trainable parameters
            reset_optimizer=False,     # reset optimizer variables with each new layer
    ):

        # Scale down the number of factors if scaling_factor is provided
        self.channels = [512, 512, 512, 512, 256, 128, 64, 32, 16, 8]
        if scaling_factor:
            assert scaling_factor > 1
            self.channels = [max(4, c // scaling_factor) for c in self.channels]

        self.batch_size = [16, 16, 16, 16, 16, 16, 8, 6, 3]
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
        self.start = True

        # Generate fized latent variables for image previews
        np.random.seed(0)
        self.z_fixed = np.random.normal(size=[self.n_examples, self.z_length])

        # Initialize placeholders
        self.x_placeholder = tf.placeholder(tf.float32, [None, None, None, 3])
        self.z_placeholder = tf.placeholder(tf.float32, [None, self.z_length])

        # Global step
        with tf.variable_scope('global_step'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.global_step_op = tf.assign(self.global_step, tf.add(self.global_step, 1))

        # Non-trainable variables for counting to next layer and incrementing value of alpha
        with tf.variable_scope('image_count'):
            self.total_imgs = tf.Variable(0.0, name='image_step', trainable=False)
            self.img_count_placeholder = tf.placeholder(tf.float32)
            self.img_step_op = tf.assign(self.total_imgs,
                tf.add(self.total_imgs, self.img_count_placeholder))

            self.img_step = tf.mod(tf.add(self.total_imgs, self.n_imgs), self.n_imgs * 2)
            self.alpha = tf.minimum(1.0, tf.div(self.img_step, self.n_imgs))
            self.layer = tf.floor_div(tf.add(self.total_imgs, self.n_imgs),  self.n_imgs * 2)

        # Initialize optimizer as member variable if not rest_optimizer, otherwise generate new
        # optimizer for each layer
        if self.reset_optimizer:
            self.lr = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
        else:
            self.g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2)
            self.d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2)

        # Initialize FeedDict
        self.feed = FeedDict(directory=img_dir)
        self.n_layers = int(np.log2(1024)) - 1
        self.networks = [self._create_network(i + 1) for i in range(self.n_layers)]

        # Initialize Session, FileWriter and Saver
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(self.logdir, graph=self.sess.graph)
        self.saver = tf.train.Saver()

        # Look in logdir to see if a saved model already exists. If so, load it
        try:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.logdir))
            print('Restored ----------------\n')
        except Exception:
            pass

    # Function for fading input of current layer into previous layer based on current value of alpha
    def _reparameterize(self, x0, x1):
        return tf.add(
            tf.scalar_mul(tf.subtract(1.0, self.alpha), x0),
            tf.scalar_mul(self.alpha, x1)
        )

    # Function for creating network layout at each layer
    def _create_network(self, layers):

        # Build the generator for this layer
        def generator(z, reuse=True):
            with tf.variable_scope('Generator', reuse=reuse):
                with tf.variable_scope('latent_vector'):
                    z = tf.expand_dims(z, 1)
                    g1 = tf.expand_dims(z, 2)
                for i in range(layers):
                    with tf.variable_scope('layer_{}'.format(i)):
                        if i > 0:
                            g1 = resize(g1)
                        if i == layers - 1 and layers > 1:
                            g0 = g1
                        with tf.variable_scope('1'):
                            if i == 0:
                                g1 = pixelwise_norm(leaky_relu(conv2d_transpose(
                                    g1, [tf.shape(g1)[0], 4, 4, self.channels[0]])))
                            else:
                                g1 = pixelwise_norm(leaky_relu(conv2d(g1, self.channels[i])))
                        with tf.variable_scope('2'):
                            g1 = pixelwise_norm(leaky_relu(conv2d(g1, self.channels[i])))
                with tf.variable_scope('rgb_layer_{}'.format(layers - 1)):
                    g1 = conv2d(g1, 3, 1, weight_norm=False)
                if layers > 1:
                    with tf.variable_scope('rgb_layer_{}'.format(layers - 2)):
                        g0 = conv2d(g0, 3, 1, weight_norm=False)
                        g = self._reparameterize(g0, g1)
                else:
                    g = g1
            return g

        # Build the discriminator for this layer
        def discriminator(x, reuse=True):
            with tf.variable_scope('Discriminator', reuse=reuse):
                if layers > 1:
                    with tf.variable_scope('rgb_layer_{}'.format(layers - 2)):
                        d0 = avg_pool(x)
                        d0 = leaky_relu(conv2d(d0, self.channels[layers - 1], 1))
                with tf.variable_scope('rgb_layer_{}'.format(layers - 1)):
                    d1 = leaky_relu(conv2d(x, self.channels[layers], 1))
                for i in reversed(range(layers)):
                    with tf.variable_scope('layer_{}'.format(i)):
                        if i == 0:
                            d1 = minibatch_stddev(d1)
                        with tf.variable_scope('1'):
                            d1 = leaky_relu(conv2d(d1, self.channels[i]))
                        with tf.variable_scope('2'):
                            if i == 0:
                                d1 = leaky_relu(conv2d(d1, self.channels[0], 4, padding='VALID'))
                            else:
                                d1 = leaky_relu(conv2d(d1, self.channels[i]))
                        if i != 0:
                            d1 = avg_pool(d1)
                        if i == layers - 1 and layers > 1:
                            d1 = self._reparameterize(d0, d1)
                with tf.variable_scope('dense'):
                    d = tf.reshape(d1, [-1, self.channels[0]])
                    d = dense_layer(d, 1)
            return d

        # image dimensions
        dim = 2 ** (layers + 1)

        # Build the current network
        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            Gz = generator(self.z_placeholder, reuse=False)
            Dz = discriminator(Gz, reuse=False)

            # Mix different resolutions of input images according to value of alpha
            with tf.variable_scope('reshape'):
                if layers > 1:
                    x0 = resize(self.x_placeholder, (dim // 2, dim // 2))
                    x0 = resize(x0, (dim, dim))
                    x1 = resize(self.x_placeholder, (dim, dim))
                    x = self._reparameterize(x0, x1)
                else:
                    x = resize(self.x_placeholder, (dim, dim))
            Dx = discriminator(x)

            # Fake and real image mixing for WGAN-GP loss function
            interp = tf.random_uniform(shape=[tf.shape(Dz)[0], 1, 1, 1], minval=0., maxval=1.)
            x_hat = interp * x + (1 - interp) * Gz
            Dx_hat = discriminator(x_hat)

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
            wd_sum = tf.summary.scalar('Wasserstein_distance_{}x{}'.format(dim, dim), wd)
            gp_sum = tf.summary.scalar('gradient_penalty_{}x{}'.format(dim, dim), gp)

        # Collecting variables to be trained by optimizers
        g_vars, d_vars = [], []
        var_scopes = ['layer_{}'.format(i) for i in range(layers)]
        var_scopes.extend(['dense', 'rgb_layer_{}'.format(layers - 1), 'rgb_layer_{}'.format(layers - 2)])
        for scope in var_scopes:
            g_vars.extend(tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope='Network/Generator/{}'.format(scope)))
            d_vars.extend(tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope='Network/Discriminator/{}'.format(scope)))

        # Generate optimizer operations
        # if self.reset_optimizer is True then initialize a new optimizer for each layer
        with tf.variable_scope('Optimize'):
            if self.reset_optimizer:
                g_train = tf.train.AdamOptimizer(
                    self.lr, self.beta1, self.beta2, name='G_optimizer_{}'.format(layers - 1)).minimize(
                    g_cost, var_list=g_vars)
                d_train = tf.train.AdamOptimizer(
                    self.lr, self.beta1, self.beta2, name='D_optimizer_{}'.format(layers - 1)).minimize(
                    d_cost, var_list=d_vars)
            else:
                g_train = self.g_optimizer.minimize(g_cost, var_list=g_vars)
                d_train = self.d_optimizer.minimize(d_cost, var_list=d_vars)

        # Print variable names to before running model
        print([var.name for var in g_vars])
        print([var.name for var in d_vars])

        # Generate preview images
        with tf.variable_scope('image_preview'):
            fake_imgs = tf.minimum(tf.maximum(Gz, -tf.ones_like(Gz)), tf.ones_like(Gz))
            real_imgs = x[:min(self.batch_size[layers - 1], 4), :, :, :]

            # Upsize images to normal visibility
            if dim < 256:
                fake_imgs = resize(fake_imgs, (256, 256))
                real_imgs = resize(real_imgs, (256, 256))

            # Concatenate images into one large image for preview, only used if 24 preview images are requested
            if self.big_image and self.n_examples == 24:
                fake_img_list = tf.unstack(fake_imgs, num=24)
                fake_img_list = [tf.concat(fake_img_list[6 * i:6 * (i + 1)], 1) for i in range(4)]
                fake_imgs = tf.concat(fake_img_list, 0)
                fake_imgs = tf.expand_dims(fake_imgs, 0)

                real_img_list = tf.unstack(real_imgs, num=min(self.batch_size[layers - 1], 4))
                real_imgs = tf.concat(real_img_list, 1)
                real_imgs = tf.expand_dims(real_imgs, 0)

            # images summaries
            fake_img_sum = tf.summary.image('fake{}x{}'.format(dim, dim),
                                            fake_imgs, self.n_examples)
            real_img_sum = tf.summary.image('real{}x{}'.format(dim, dim),
                                            real_imgs, 4)

        return (dim, wd, gp, wd_sum, gp_sum, g_train, d_train,
                fake_img_sum, real_img_sum, Gz)

    # Summary adding function
    def _add_summary(self, string, gs):
        self.writer.add_summary(string, gs)

    # Latent variable 'z' generator
    def _z(self, batch_size):
        return np.random.normal(0.0, 1.0, [batch_size, self.z_length])

    # Main training function
    def train(self):
        prev_layer = None
        start_time = dt.datetime.now()
        total_imgs = self.sess.run(self.total_imgs)

        while total_imgs < (self.n_layers - 0.5) * self.n_imgs * 2:

            # Get current layer, global step, alpha and total number of images used so far
            layer, gs, img_step, alpha, total_imgs = self.sess.run([
                self.layer, self.global_step, self.img_step, self.alpha, self.total_imgs])
            layer = int(layer)

            # Global step interval to save model and generate image previews
            save_interval = max(1000, 10000 // 2 ** layer)

            # Get network operations and loss functions for current layer
            (dim, wd, gp, wd_sum, gp_sum, g_train, d_train,
             fake_img_sum, real_img_sum, Gz) = self.networks[layer]

            # Get training data and latent variables to store in feed_dict
            feed_dict = {self.x_placeholder: self.feed.next_batch(self.batch_size[layer], dim),
                         self.z_placeholder: self._z(self.batch_size[layer])}

            # Reset start times if a new layer has begun training
            if layer != prev_layer:
                start_time = dt.datetime.now()

            # Here's where we actually train the model
            for _ in range(self.batch_repeats):
                self.sess.run(g_train, feed_dict)
                self.sess.run(d_train, feed_dict)

            # Get loss values and summaries
            wd_, gp_, wd_sum_str, gp_sum_str = self.sess.run([wd, gp, wd_sum, gp_sum], feed_dict)

            # Print current status, loss functions, etc.
            percent_done = np.round(img_step * 50 / self.n_imgs, 4)
            imgs_done = int(img_step)
            cur_layer_imgs = self.n_imgs * 2
            if dim == 4:
                percent_done = np.round((percent_done - 50) * 2, 4)
                imgs_done -= self.n_imgs
                cur_layer_imgs //= 2
            print('dimensions: {}x{} ---- {}% ---- images: {}/{} ---- alpha: {} ---- global step: {}'
                  '\nWasserstein distance: {}\ngradient penalty: {}\n'.format(
                dim, dim, percent_done, imgs_done, cur_layer_imgs, alpha, gs, wd_, gp_))

            # Log scalar data every 20 global steps
            if gs % 20 == 0:
                self._add_summary(wd_sum_str, gs)
                self._add_summary(gp_sum_str, gs)

            # Operations to run every save interval
            if gs % save_interval == 0:

                # Do not save the model or generate images immediately after loading/preloading
                if self.start:
                    self.start = False

                # Save the model and generate image previews
                else:
                    print('saving and making images...\n')
                    self.saver.save(
                        self.sess, os.path.join(self.logdir, "model.ckpt"),
                        global_step=self.global_step)
                    real_img_sum_str = self.sess.run(real_img_sum, feed_dict)
                    img_preview_feed_dict = {
                        self.x_placeholder: feed_dict[self.x_placeholder][:4],
                        self.z_placeholder: self.z_fixed}
                    fake_img_sum_str = self.sess.run(fake_img_sum, img_preview_feed_dict)
                    self._add_summary(fake_img_sum_str, gs)
                    self._add_summary(real_img_sum_str, gs)

            # Increment image count and global step variables
            img_count = self.batch_repeats * self.batch_size[layer]
            self.sess.run(self.global_step_op)
            self.sess.run(self.img_step_op, {self.img_count_placeholder: img_count})

            # Calculate and print estimated time remaining
            prev_layer = layer
            avg_time = (dt.datetime.now() - start_time) / (imgs_done + self.batch_size[layer])
            steps_remaining = cur_layer_imgs - imgs_done
            time_reamining = avg_time * steps_remaining
            print('est. time remaining on current layer: {}'.format(time_reamining))


    # Function for generating images from a 1D or 2D array of latent vectors
    def generate(self, z):
        if len(z.shape) == 1:
            z = np.expand_dims(z, 0)

        cur_layer = int(self.sess.run(self.layer))
        G = self.networks[cur_layer][9]
        imgs = self.sess.run(G, {self.z_placeholder: z})

        imgs = np.minimum(imgs, 1.0)
        imgs = np.maximum(imgs, -1.0)
        imgs = (imgs + 1) * 255 / 2
        imgs = np.uint8(imgs)

        if len(imgs.shape) == 4:
            imgs = np.squeeze(imgs, 0)
        return imgs


if __name__ == '__main__':
    progan = ProGAN(
        logdir='logdir_v2',
        img_dir='img_arrays',
    )
    progan.train()