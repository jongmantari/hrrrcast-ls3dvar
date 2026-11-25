# ls3dvar/hrrrcast/encoder.py  (extracted from mantari/encoder_runtime.py)
from __future__ import annotations

import tensorflow as tf


class EncoderRuntime:

    def __init__(self, model):

        self.encoder = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(
                "encoder_2_maxpooling"
            ).output,
        )

    def encode(
        self,
        x,
    ):

        z = self.encoder(
            x,
            training=False,
        )

        return z.numpy()

    def latent_shape(self):

        return self.encoder.output_shape
