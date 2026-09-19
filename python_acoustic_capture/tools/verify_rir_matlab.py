"""Prepare deterministic cases for verify_rir_matlab.m; no audio hardware used."""
from pathlib import Path
import argparse
import numpy as np
from scipy.io import savemat
from scipy.signal import fftconvolve
from acoustic_capture.signals import exponential_sweep
from acoustic_capture.deconvolution import estimate_impulse_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    root = parser.parse_args().output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    fs, post = 48000, .5
    for index, (duration, level) in enumerate([(.5, -12), (8, -18), (8, -30), (8, -60)], 1):
        sweep = exponential_sweep(fs, 40, 22000, duration, level)
        played = np.pad(sweep, (0, round(post * fs)))
        impulse = np.zeros((round(post * fs), 2))
        for channel, (delay, gain) in enumerate([(173, .8), (180, .65)]):
            impulse[delay, channel] = gain
            impulse[delay + 121, channel] = .2 * gain
            impulse[delay + 367, channel] = -.1 * gain
        response = np.column_stack([
            fftconvolve(played, impulse[:, channel])[:len(played)] for channel in range(2)
        ])
        estimated = estimate_impulse_response(sweep, response, round(post * fs))
        savemat(root / f"case_{index}.mat", dict(
            fs=float(fs), post=post, duration=float(duration), level=float(level),
            sweep=sweep[:, None], response=response, python_ir=estimated,
        ))
    print(root)


if __name__ == "__main__":
    main()
