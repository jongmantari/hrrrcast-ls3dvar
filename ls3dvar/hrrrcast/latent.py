# ls3dvar/hrrrcast/latent.py  (extracted from mantari/ls3dvar/latent_extractor.py)
import tensorflow as tf


def extract_latent(
    model,
    X,
):
    probe = tf.keras.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(
                "processed_inputs"
            ).output,
            model.get_layer(
                "encoder_0_maxpooling"
            ).output,
            model.get_layer(
                "encoder_1_maxpooling"
            ).output,
            model.get_layer(
                "encoder_2_maxpooling"
            ).output,
            model.get_layer(
                "time_bn_relu_relu"
            ).output,
            model.get_layer(
                "diff_inputs_scaled"
            ).output,
            model.get_layer(
                "noised_input_slice"
            ).output,
        ],
    )

    (
        processed,
        skip0,
        skip1,
        latent,
        time_vec,
        diff_scaled,
        noised,
    ) = probe(
        X,
        training=False,
    )

    state = {
        "processed": processed,
        "skip0": skip0,
        "skip1": skip1,
        "time_vec": time_vec,
        "diff_inputs_scaled": diff_scaled,
        "noised_input_slice": noised,
    }

    return latent, state
