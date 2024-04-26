import preprocessing

import tensorflow as tf
import numpy as np


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
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(230,))
        self.dense1 = tf.keras.layers.Dense(256)
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(128)
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(1, activation=activation_name)

        self.activation = activation_name

    def call(self, inputs, training=False):
        x = self.input_layer(inputs)
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
        if self.activation == "softmax":
            loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            )
        else:
            loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))

        return loss


def get_data(split=0.8):
    inputs, labels = preprocessing.process_data()

    assert len(inputs) == len(labels), "Input and label lengths do not match"

    inputs = np.asarray(inputs).astype(np.float32)
    labels = np.asarray(labels).astype(np.float32)

    split_idx = int(len(inputs) * split)

    X_train, X_val = inputs[:split_idx], inputs[split_idx:]
    y_train, y_val = labels[:split_idx], labels[split_idx:]
    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = get_data()

    model = model("sigmoid")

    optimizer = tf.keras.optimizers.legacy.Adam()
    model.compile(optimizer=optimizer, loss=model.loss)

    model.fit(X_train, y_train, epochs=10, batch_size=32)
