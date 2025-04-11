from fractions import Fraction
from pathlib import Path

import av
from PIL import Image


def save_video(path: Path, images: list[Image.Image], fps: int | Fraction = 24):
    container = av.open(path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "veryfast"}

    for ii, img in enumerate(images):
        if ii == 0:
            stream.height = img.height
            stream.width = img.width

        frame = av.VideoFrame.from_image(img)
        for packet in stream.encode(frame):
            container.mux(packet)

    # Finalize the file
    for packet in stream.encode():
        container.mux(packet)

    container.close()
