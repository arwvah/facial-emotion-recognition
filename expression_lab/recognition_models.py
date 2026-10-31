"""Pinned optional recognition weights; downloads happen only during explicit setup."""

from .backends import ModelSpec
from .emotion import ROOT, download_asset, download_detector

NANODET_PATH = ROOT / "models" / "nanodet.onnx"
NANODET_SHA256 = "4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186"
NANODET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    "48b10fa505aeb51fbd127fc60a5c660040e823b7/"
    "models/object_detection_nanodet/object_detection_nanodet_2022nov.onnx"
)
HSEMOTION_PATH = ROOT / "models" / "enet_b0_8_best_vgaf.onnx"
HSEMOTION_SHA256 = "fa07e841fd06c7a67ee651ea4e6e4a3a2bb5695f47b37a7da50492526f59c898"
HSEMOTION_URL = (
    "https://raw.githubusercontent.com/av-savchenko/face-emotion-recognition/"
    "520a051c64cd191521e5934655314e769a319684/"
    "models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx"
)


def hsemotion_spec():
    return ModelSpec(
        HSEMOTION_PATH,
        HSEMOTION_SHA256,
        labels=("angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"),
        width=224,
        height=224,
        color="rgb",
        scale=1 / 255,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation="linear",
    )


def setup_recognition():
    return [
        download_detector(),
        download_asset(NANODET_PATH, NANODET_URL, NANODET_SHA256, 3_800_954, "NanoDet Plus"),
        download_asset(
            HSEMOTION_PATH, HSEMOTION_URL, HSEMOTION_SHA256, 16_039_595, "HSEmotion RGB CNN"
        ),
    ]
