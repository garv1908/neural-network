import tensorflow as tf

def train(self, epochs):
        self.model = tf.keras.models.Sequential()

        self.model.add(tf.keras.layers.Convolution2D(
            input_shape = (self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_CHANNELS),
            kernel_size = 5,
            filters = 8,
            strides = 1,
            activation = tf.keras.activations.relu,
            kernel_initializer = tf.keras.initializers.VarianceScaling()
        ))

        self.model.add(tf.keras.layers.MaxPooling2D(
            pool_size = (2, 2),
            strides = (2, 2),
        ))

        self.model.add(tf.keras.layers.Convolution2D(
            kernel_size = 5,
            filters = 16,
            strides = 1,
            activation = tf.keras.activations.relu,
            kernel_initializer = tf.keras.initializers.VarianceScaling()
        ))

        self.model.add(tf.keras.layers.MaxPooling2D(
            pool_size = (2, 2),
            strides = (2, 2),
        ))

        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(
            units = 128,
            activation = tf.keras.activations.relu
        ))

        self.model.add(tf.keras.layers.Dropout(0.2))

        self.model.add(tf.keras.layers.Dense(
            units = 10,
            activation = tf.keras.activations.softmax,
            kernel_initializer = tf.keras.initializers.VarianceScaling()
        ))

        self.model.summary()

        adam_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

        self.model.compile(
            optimizer = adam_optimizer,
            loss = tf.keras.losses.categorical_crossentropy,
            metrics = ['accuracy']
        )

        training_history = self.model.fit(
            self.images,
            self.labels,
            epochs=epochs
        )

        self.save_model()
