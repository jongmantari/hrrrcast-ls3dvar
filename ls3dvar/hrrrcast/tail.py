# ls3dvar/hrrrcast/tail.py  (extracted from mantari/tail_model_runtime_tf.py)
from __future__ import annotations

import tensorflow as tf


class TailModelRuntimeTF:
    """
    TensorFlow-native HRRRCast tail.

    Fully differentiable.

    Produces:

        output_refine
            +
        diff_inputs_scaled
            +
        noised_input_slice

            ↓

        shortcut_output

            ↓

        unpadded_output
    """

    def __init__(
        self,
        crop_top=3,
        crop_bottom=2,
        crop_left=1,
        crop_right=0,
    ):

        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.crop_left = crop_left
        self.crop_right = crop_right

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    def validate(
        self,
        output_refine,
        diff_inputs_scaled,
        noised_input_slice,
    ):

        if output_refine.shape != diff_inputs_scaled.shape:

            raise ValueError(
                f"output_refine {output_refine.shape} "
                f"!= diff_inputs_scaled "
                f"{diff_inputs_scaled.shape}"
            )

        if output_refine.shape != noised_input_slice.shape:

            raise ValueError(
                f"output_refine {output_refine.shape} "
                f"!= noised_input_slice "
                f"{noised_input_slice.shape}"
            )

    # -------------------------------------------------
    # Shortcut
    # -------------------------------------------------

    def shortcut(
        self,
        output_refine,
        diff_inputs_scaled,
        noised_input_slice,
    ):

        self.validate(
            output_refine,
            diff_inputs_scaled,
            noised_input_slice,
        )

        shortcut = (
            output_refine
            + diff_inputs_scaled
            + noised_input_slice
        )

        return shortcut

    # -------------------------------------------------
    # Unpadding
    # -------------------------------------------------

    def unpad(
        self,
        shortcut,
    ):

        h = tf.shape(shortcut)[1]
        w = tf.shape(shortcut)[2]

        y0 = self.crop_top

        y1 = (
            h
            if self.crop_bottom == 0
            else h - self.crop_bottom
        )

        x0 = self.crop_left

        x1 = (
            w
            if self.crop_right == 0
            else w - self.crop_right
        )

        return shortcut[
            :,
            y0:y1,
            x0:x1,
            :
        ]

    # -------------------------------------------------
    # Main forecast
    # -------------------------------------------------

    def forecast(
        self,
        output_refine,
        diff_inputs_scaled,
        noised_input_slice,
    ):

        shortcut = self.shortcut(
            output_refine,
            diff_inputs_scaled,
            noised_input_slice,
        )

        forecast = self.unpad(
            shortcut
        )

        return forecast

    # -------------------------------------------------
    # Diagnostics version
    # -------------------------------------------------

    def forecast_with_intermediates(
        self,
        output_refine,
        diff_inputs_scaled,
        noised_input_slice,
    ):

        shortcut = self.shortcut(
            output_refine,
            diff_inputs_scaled,
            noised_input_slice,
        )

        forecast = self.unpad(
            shortcut
        )

        return {
            "shortcut": shortcut,
            "forecast": forecast,
        }
