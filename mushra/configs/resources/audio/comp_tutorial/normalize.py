import soundfile as sf
import numpy as np
from glob import glob
import os
import pyloudnorm

m = pyloudnorm.Meter(44100)

fns = []
xs = []
for fn in glob(os.path.join(os.path.dirname(__file__), "*.ogg")):
    if "source" not in os.path.basename(fn):
        x, sr = sf.read(fn, always_2d=True)
        x -= np.mean(x, axis=0, keepdims=True)
        l = m.integrated_loudness(x)
        x *= 10 ** (-l / 20)
        fns.append(fn)
        xs.append(x)
max_peak = max(np.max(np.abs(_x)) for _x in xs)
for fn, x in zip(fns, xs):
    x /= max_peak
    sf.write(fn[:-4]+".wav", x, sr, "FLOAT")

# convert to ogg using windows cmd command:
# for %f in (*.wav) do ffmpeg -i "%f" -qscale:a 7 -y "%~nf.ogg" && del "%f"