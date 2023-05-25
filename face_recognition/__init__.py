# -*- coding: utf-8 -*-

__author__ = """Adam Geitgey"""
__email__ = "ageitgey@gmail.com"
__version__ = "1.4.0"

from .api import (
    load_image_file,  # pyright: ignore[reportUnknownVariableType]
    face_locations,  # pyright: ignore[reportUnknownVariableType]
    batch_face_locations,  # pyright: ignore[reportUnknownVariableType]
    face_landmarks,  # pyright: ignore[reportUnknownVariableType]
    face_encodings,  # pyright: ignore[reportUnknownVariableType]
    compare_faces,  # pyright: ignore[reportUnknownVariableType]
    face_distance,  # pyright: ignore[reportUnknownVariableType]
)
