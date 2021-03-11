import tensorflow as tf


def get_model(path=None):
    """Define and compile the tf model"""
    neurons = 64
    input = tf.keras.layers.Input(4)

    X = tf.keras.layers.Dense(neurons, "relu")(input)
    X = tf.keras.layers.Dense(neurons, "relu")(X)
    X = tf.keras.layers.Dense(neurons, "relu")(X)

    output = tf.keras.layers.Dense(1, "sigmoid", name="output")(X)

    agent = tf.keras.models.Model(inputs=[input], outputs=[output])

    if path:
        agent.load_weights(path)

    return agent

