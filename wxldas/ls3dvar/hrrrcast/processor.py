# ls3dvar/hrrrcast/processor.py  (extracted from mantari/processor_runtime_tf.py)
from __future__ import annotations

import tensorflow as tf


class ProcessorRuntimeTF:

    def __init__(self, model):

        self.blocks = []

        for i in range(14):

            self.blocks.append({

                "conv1":
                model.get_layer(
                    f"processor_{i}_res_{i+1}_1_conv"
                ),

                "ln1":
                model.get_layer(
                    f"processor_{i}_res_{i+1}_1_lnorm"
                ),

                "conv2":
                model.get_layer(
                    f"processor_{i}_res_{i+1}_2_conv"
                ),

                "ln2":
                model.get_layer(
                    f"processor_{i}_res_{i+1}_2_lnorm"
                ),

                "film_gamma":
                model.get_layer(
                    f"processor_{i}_res_{i+1}_film_gamma_dense"
                ),

                "film_beta":
                model.get_layer(
                    f"processor_{i}_res_{i+1}_film_beta_dense"
                ),

                "cbam_ch_mlp1":
                model.get_layer(
                    f"processor_{i}_res_{i+1}_cbam_ch_mlp_1"
                ),

                "cbam_ch_mlp2":
                model.get_layer(
                    f"processor_{i}_res_{i+1}_cbam_ch_mlp_2"
                ),

                "cbam_sp_conv":
                model.get_layer(
                    f"processor_{i}_res_{i+1}_cbam_sp_conv7"
                ),
            })

    def run_block(
        self,
        x,
        time_vec,
        block,
    ):

        shortcut = x

        x = block["conv1"](x)
        x = block["ln1"](x)
        x = tf.nn.relu(x)

        x = block["conv2"](x)
        x = block["ln2"](x)

        gamma = block[
            "film_gamma"
        ](time_vec)

        beta = block[
            "film_beta"
        ](time_vec)

        gamma = tf.reshape(
            gamma,
            (-1,1,1,256)
        )

        beta = tf.reshape(
            beta,
            (-1,1,1,256)
        )

        x = x * gamma + beta

        ch = tf.reduce_mean(
            x,
            axis=(1,2)
        )

        ch = block[
            "cbam_ch_mlp1"
        ](ch)

        ch = block[
            "cbam_ch_mlp2"
        ](ch)

        ch = tf.reshape(
            ch,
            (-1,1,1,256)
        )

        x = x * ch

        sp = tf.reduce_mean(
            x,
            axis=-1,
            keepdims=True
        )

        sp = block[
            "cbam_sp_conv"
        ](sp)

        sp = tf.sigmoid(sp)

        x = x * sp

        x = x + shortcut

        x = tf.nn.relu(x)

        return x

    def run(
        self,
        latent,
        time_vec,
    ):

        x = latent

        for block in self.blocks:

            x = self.run_block(
                x,
                time_vec,
                block
            )

        return x
