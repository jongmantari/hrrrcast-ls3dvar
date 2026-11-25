#!/usr/bin/env python3
# ls3dvar/tests/test_decode_parity.py  (extracted from mantari/workflows/test_decode_parity.py)
"""
Encode -> process -> decode parity test (NO forecast tail).

Runs the standalone runtimes

    EncoderRuntime -> ProcessorRuntimeTF -> DecoderRuntime

and compares every stage against taps from a single forward pass of the
original Keras model:

    latent (encoder_2_maxpooling)
    proc_out (processor_13_res_14_relu)
    d0, d1, d2 (decoder_N_recomp)
    refine (output_refine_recomp)

Optionally compares against / saves npz files.

Run:
    python -m ls3dvar.tests.test_decode_parity
    python -m ls3dvar.tests.test_decode_parity --save /tmp/decode_out.npz
    python -m ls3dvar.tests.test_decode_parity --ref-npz ref.npz --ref-map refine=some_key
"""
from ls3dvar.paths import DEFAULT_CONFIG as TEST_CONFIG, MODEL_PATH, SRC_DIR
import argparse
import sys

sys.path.insert(0, str(SRC_DIR))

import yaml
import numpy as np
import tensorflow as tf
import resnet

from ls3dvar.hrrrcast.encoder import EncoderRuntime
from ls3dvar.hrrrcast.processor import ProcessorRuntimeTF
from ls3dvar.hrrrcast.decoder_numpy import DecoderRuntime

CUSTOM_OBJECTS = {
    "CastLayer": resnet.CastLayer,
    "ChannelPoolAvg": resnet.ChannelPoolAvg,
    "ChannelSliceLayer": resnet.ChannelSliceLayer,
    "OutputMaskLayer": resnet.OutputMaskLayer,
    "RecomputeSubModel": resnet.RecomputeSubModel,
    "ReflectPadLayer": resnet.ReflectPadLayer,
    "SpatialGroupedLayer": resnet.SpatialGroupedLayer,
    "TimeCondLayer": resnet.TimeCondLayer,
    "UnpadLayer": resnet.UnpadLayer,
}

# Layer names taken from --list-layers of the model
ENCODER_OUT = "encoder_2_maxpooling"      # latent
SKIP0_LAYER = "encoder_0_maxpooling"      # (136, 124, 288) at this input size
SKIP1_LAYER = "encoder_1_maxpooling"      # (68, 62, 256)
PROCESSED_LAYER = "processed_inputs"      # (272, 248, 323)
TIME_VEC_LAYER = "time_bn_relu_relu"      # (256,)
PROC_OUT_LAYER = "processor_13_res_14_relu"


def load_config():
    with open(TEST_CONFIG) as f:
        return yaml.safe_load(f)


def build_operational_input(hrrr_npz, gfs_npz):
    hrrr = np.load(hrrr_npz, allow_pickle=True)
    gfs = np.load(gfs_npz, allow_pickle=True)

    model_input_hrrr = hrrr["model_input"]
    model_input_gfs = gfs["model_input"]

    predicted_channels = (
        len(hrrr["pl_vars"]) * len(hrrr["levels"])
        + len(hrrr["sfc_vars"])
    )

    nlat = model_input_hrrr.shape[1]
    nlon = model_input_hrrr.shape[2]

    rand_channel = np.ones((1, nlat, nlon, predicted_channels), dtype=np.float32)
    date_channel = np.ones((1, nlat, nlon, 7), dtype=np.float32)
    step_channel = np.ones((1, nlat, nlon, 1), dtype=np.float32)
    lead_channel = np.ones((1, nlat, nlon, 1), dtype=np.float32)

    X = np.concatenate(
        [
            model_input_hrrr[:, :, :, :predicted_channels],
            model_input_gfs[0:1, :, :, :],
            rand_channel,
            model_input_hrrr[:, :, :, predicted_channels:],
            date_channel,
            step_channel,
            lead_channel,
        ],
        axis=-1,
    )
    return X.astype(np.float32), predicted_channels


def to_np(t):
    return np.asarray(t.numpy() if hasattr(t, "numpy") else t, dtype=np.float32)


def compare(name, got, ref, atol, rtol):
    got, ref = to_np(got), to_np(ref)
    if got.shape != ref.shape:
        print(f"FAIL  {name:12s} shape {got.shape} != {ref.shape}")
        return False
    diff = np.abs(got - ref)
    denom = np.linalg.norm(ref.ravel()) + 1e-12
    rel_l2 = np.linalg.norm((got - ref).ravel()) / denom
    ok = bool(np.allclose(got, ref, atol=atol, rtol=rtol))
    print(
        f"{'PASS' if ok else 'FAIL'}  {name:12s} shape={got.shape} "
        f"max_abs={diff.max():.3e} mean_abs={diff.mean():.3e} rel_l2={rel_l2:.3e}"
    )
    return ok


def banner(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--save", default=None,
                    help="write runtime outputs to this npz")
    ap.add_argument("--ref-npz", default=None,
                    help="optional npz to compare runtime outputs against")
    ap.add_argument("--ref-map", action="append", default=[],
                    help="name=key_in_ref_npz (repeatable); "
                         "names: latent proc_out d0 d1 d2 refine")
    args = ap.parse_args()

    banner("LOAD MODEL")
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
        safe_mode=False,
    )

    cfg = load_config()
    X, predicted_channels = build_operational_input(cfg["hrrr_npz"], cfg["gfs_npz"])
    print("Input shape:", X.shape)
    print("Predicted channels (pl*levels + sfc):", predicted_channels)

    L = model.get_layer
    dec0, dec1, dec2 = L("decoder_0_recomp"), L("decoder_1_recomp"), L("decoder_2_recomp")
    refine_layer = L("output_refine_recomp")

    # ------------------------------------------------------------------
    # 1. Reference: one forward pass of the original model, all taps
    # ------------------------------------------------------------------
    banner("REFERENCE FORWARD PASS (original model)")
    taps = {
        "latent": L(ENCODER_OUT).output,
        "skip0": L(SKIP0_LAYER).output,
        "skip1": L(SKIP1_LAYER).output,
        "processed": L(PROCESSED_LAYER).output,
        "time_vec": L(TIME_VEC_LAYER).output,
        "proc_out": L(PROC_OUT_LAYER).output,
        "in0": dec0.input[0],   # concat(up(proc_out), skip1)
        "in1": dec1.input[0],   # concat(up(d0), skip0)
        "in2": dec2.input[0],   # concat(up(d1), processed)
        "d0": dec0.output,
        "d1": dec1.output,
        "d2": dec2.output,
        "refine": refine_layer.output,
    }
    ref_model = tf.keras.Model(inputs=model.input, outputs=list(taps.values()))
    ref = dict(zip(taps.keys(), [to_np(t) for t in ref_model(X, training=False)]))
    for k, v in ref.items():
        print(f"  {k:10s} {v.shape}")

    results = []

    # Sanity check: the skip taps chosen by layer name must equal the skips
    # actually fed into the decoder concats (concat order: [features, skip]).
    banner("TAP SANITY (named skip layers vs slices of decoder concat inputs)")
    sliced = {
        "skip1": ref["in0"][..., ref["proc_out"].shape[-1]:],
        "skip0": ref["in1"][..., ref["d0"].shape[-1]:],
        "processed": ref["in2"][..., ref["d1"].shape[-1]:],
    }
    for k, v in sliced.items():
        results.append(compare(f"tap:{k}", ref[k], v, args.atol, args.rtol))

    # ------------------------------------------------------------------
    # 2. Runtime path: encoder -> processor -> decoder
    # ------------------------------------------------------------------
    banner("RUNTIME: ENCODER")
    latent = EncoderRuntime(model).encode(X)
    print("latent:", latent.shape)

    # skips / processed / time_vec come from the encoder side of the model.
    skip_model = tf.keras.Model(
        inputs=model.input,
        outputs=[taps["skip0"], taps["skip1"], taps["processed"], taps["time_vec"]],
    )
    skip0, skip1, processed, time_vec = [
        to_np(t) for t in skip_model(X, training=False)
    ]

    banner("RUNTIME: PROCESSOR")
    proc_out = to_np(
        ProcessorRuntimeTF(model).run(tf.constant(latent), tf.constant(time_vec))
    )
    print("proc_out:", proc_out.shape)

    banner("RUNTIME: DECODER")
    dec_out = DecoderRuntime(model).decode_features(
        tf.constant(proc_out),
        tf.constant(skip1),
        tf.constant(skip0),
        tf.constant(processed),
        tf.constant(time_vec),
    )
    outputs = {
        "latent": latent,
        "proc_out": proc_out,
        "d0": to_np(dec_out["d0"]),
        "d1": to_np(dec_out["d1"]),
        "d2": to_np(dec_out["d2"]),
        "refine": to_np(dec_out["refine"]),
    }

    # ------------------------------------------------------------------
    # 3. Parity vs reference model
    # ------------------------------------------------------------------
    banner("PARITY: runtime vs original model")
    for name in ["latent", "proc_out", "d0", "d1", "d2", "refine"]:
        results.append(compare(name, outputs[name], ref[name], args.atol, args.rtol))

    if outputs["refine"].shape[-1] != predicted_channels:
        print(
            f"INFO  refine has {outputs['refine'].shape[-1]} channels, "
            f"predicted_channels={predicted_channels}"
        )

    # ------------------------------------------------------------------
    # 4. Optional: save / compare against an external npz
    # ------------------------------------------------------------------
    if args.save:
        np.savez_compressed(args.save, **outputs)
        print(f"\nSaved runtime outputs -> {args.save}")

    if args.ref_npz:
        banner(f"PARITY: runtime vs {args.ref_npz}")
        ref_npz = np.load(args.ref_npz, allow_pickle=True)
        print("npz keys:", {k: ref_npz[k].shape for k in ref_npz.files
                            if hasattr(ref_npz[k], "shape")})
        mapping = dict(m.split("=", 1) for m in args.ref_map)
        if not mapping:
            mapping = {k: k for k in outputs if k in ref_npz.files}
        if not mapping:
            print("No matching keys. Pass --ref-map name=key (names: "
                  + " ".join(outputs) + ")")
        for name, key in mapping.items():
            if name not in outputs or key not in ref_npz.files:
                print(f"SKIP  {name}={key} (not found)")
                continue
            results.append(
                compare(f"npz:{name}", outputs[name], ref_npz[key], args.atol, args.rtol)
            )

    banner("SUMMARY")
    print("ALL PASS" if all(results) else "SOME FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
