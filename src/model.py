import tensorflow as tf


class model(tf.keras.Model):
    """
    The general model consists of a four-layer network
    as depicted in figure 2:
    -An input layer of 230 nodes
    -A dense hidden layer of 256 neurons
    -A second dense hidden layer of 128 neurons
    -A dense output layer, with activation dependent on
    the type of target variable the model is being asked to estimate.
    Each dense layer, aside from the output layer, uses rectilinear units (relu)
    for nonlinear activation.
    Batch normalization and dropout are employed between each dense layer
    to prevent overfitting and improve performance
    when generalizing with out-of-sample data
    """

    def __init__(self, activation_name):
        super(model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256)
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(128)
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(1, activation=activation_name)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.batch_norm1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        return x
    
    def loss(self, y_pred, y_true):
        if self.dense3.activation == 'softmax':
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))

        else: loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))

        return loss
