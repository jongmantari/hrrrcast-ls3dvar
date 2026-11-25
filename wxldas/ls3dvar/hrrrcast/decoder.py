# ls3dvar/hrrrcast/decoder.py  (extracted from mantari/decoder_runtime_tf.py)
from __future__ import annotations

import tensorflow as tf


class DecoderRuntimeTF:

    def __init__(self, model):

        self.dec0 = model.get_layer(
            "decoder_0_recomp"
        ).submodel

        self.dec1 = model.get_layer(
            "decoder_1_recomp"
        ).submodel

        self.dec2 = model.get_layer(
            "decoder_2_recomp"
        ).submodel

        self.output_refine = model.get_layer(
            "output_refine_recomp"
        ).submodel

    def decode_features(
        self,
        processor_output,
        skip1,
        skip0,
        processed,
        time_vec,
    ):

        x = tf.image.resize(
            processor_output,
            (
                skip1.shape[1],
                skip1.shape[2],
            ),
            method="nearest",
        )

        x = tf.concat(
            [x, skip1],
            axis=-1,
        )

        d0 = self.dec0(
            [x, time_vec],
            training=False,
        )

        x = tf.image.resize(
            d0,
            (
                skip0.shape[1],
                skip0.shape[2],
            ),
            method="nearest",
        )

        x = tf.concat(
            [x, skip0],
            axis=-1,
        )

        d1 = self.dec1(
            [x, time_vec],
            training=False,
        )

        x = tf.image.resize(
            d1,
            (
                processed.shape[1],
                processed.shape[2],
            ),
            method="nearest",
        )

        x = tf.concat(
            [x, processed],
            axis=-1,
        )

        d2 = self.dec2(
            [x, time_vec],
            training=False,
        )

        refine = self.output_refine(
            [d2, time_vec],
            training=False,
        )

        return {

            "d0": d0,

            "d1": d1,

            "d2": d2,

            "refine": refine,
        }
