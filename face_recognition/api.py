# -*- coding: utf-8 -*-

import PIL.Image
import dlib
import numpy as np
from PIL import ImageFile

try:
    import face_recognition_models
except Exception:
    print(
        "Please install `face_recognition_models` with this command before using `face_recognition`:\n"
    )
    print("pip install git+https://github.com/ageitgey/face_recognition_models")
    quit()

ImageFile.LOAD_TRUNCATED_IMAGES = True

face_detector = (  # pyright: ignore[reportUnknownVariableType]
    dlib.get_frontal_face_detector()  # pyright: ignore[reportUnknownMemberType]
)

predictor_68_point_model = (  # pyright: ignore[reportUnknownVariableType]
    face_recognition_models.pose_predictor_model_location()  # pyright: ignore[reportUnknownMemberType]
)
pose_predictor_68_point = dlib.shape_predictor(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    predictor_68_point_model,
)

predictor_5_point_model = (  # pyright: ignore[reportUnknownVariableType]
    face_recognition_models.pose_predictor_five_point_model_location()  # pyright: ignore[reportUnknownMemberType]  # pyright: ignore[reportUnknownMemberType]
)
pose_predictor_5_point = dlib.shape_predictor(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    predictor_5_point_model,
)

cnn_face_detection_model = (  # pyright: ignore[reportUnknownVariableType]
    face_recognition_models.cnn_face_detector_model_location()  # pyright: ignore[reportUnknownMemberType]
)
cnn_face_detector = dlib.cnn_face_detection_model_v1(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    cnn_face_detection_model,
)

face_recognition_model = (  # pyright: ignore[reportUnknownVariableType]
    face_recognition_models.face_recognition_model_location()  # pyright: ignore[reportUnknownMemberType]
)
face_encoder = dlib.face_recognition_model_v1(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    face_recognition_model,
)


def _rect_to_css(  # pyright: ignore[reportUnknownParameterType]
    rect,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order

    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return (
        rect.top(),  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        rect.right(),  # pyright: ignore[reportUnknownMemberType]
        rect.bottom(),  # pyright: ignore[reportUnknownMemberType]
        rect.left(),  # pyright: ignore[reportUnknownMemberType]
    )


def _css_to_rect(  # pyright: ignore[reportUnknownParameterType]
    css,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        css[3],
        css[0],
        css[1],
        css[2],
    )


def _trim_css_to_bounds(  # pyright: ignore[reportUnknownParameterType]
    css,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    image_shape,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return (
        max(  # pyright: ignore[reportUnknownVariableType]
            css[0],  # pyright: ignore[reportUnknownArgumentType]
            0,
        ),
        min(
            css[1],  # pyright: ignore[reportUnknownArgumentType]
            image_shape[1],  # pyright: ignore[reportUnknownArgumentType]
        ),
        min(
            css[2],  # pyright: ignore[reportUnknownArgumentType]
            image_shape[0],  # pyright: ignore[reportUnknownArgumentType]
        ),
        max(
            css[3],  # pyright: ignore[reportUnknownArgumentType]
            0,
        ),
    )


def face_distance(
    face_encodings,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    face_to_compare,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param face_encodings: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if (
        len(
            face_encodings,  # pyright: ignore[reportUnknownArgumentType]
        )
        == 0
    ):
        return np.empty((0))

    return np.linalg.norm(
        face_encodings - face_to_compare,  # pyright: ignore[reportUnknownArgumentType]
        axis=1,
    )


def load_image_file(
    file,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    mode="RGB",  # pyright: ignore[reportMissingParameterType]
):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array

    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    im = PIL.Image.open(
        file,  # pyright: ignore[reportUnknownArgumentType]
    )
    if mode:
        im = im.convert(mode)
    return np.array(im)


def _raw_face_locations(  # pyright: ignore[reportUnknownParameterType]
    img,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    number_of_times_to_upsample=1,  # pyright: ignore[reportMissingParameterType]
    model="hog",  # pyright: ignore[reportMissingParameterType]
):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of dlib 'rect' objects of found face locations
    """
    if model == "cnn":
        return cnn_face_detector(  # pyright: ignore[reportUnknownVariableType]
            img,
            number_of_times_to_upsample,
        )
    else:
        return face_detector(  # pyright: ignore[reportUnknownVariableType]
            img,
            number_of_times_to_upsample,
        )


def face_locations(  # pyright: ignore[reportUnknownParameterType]
    img,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    number_of_times_to_upsample=1,  # pyright: ignore[reportMissingParameterType]
    model="hog",  # pyright: ignore[reportMissingParameterType]
):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                  deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of tuples of found face locations in css (top, right, bottom, left) order
    """
    if model == "cnn":
        return [  # pyright: ignore[reportUnknownVariableType]
            _trim_css_to_bounds(
                _rect_to_css(
                    face.rect,  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
                ),
                img.shape,  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
            )
            for face in _raw_face_locations(  # pyright: ignore[reportUnknownVariableType]
                img,  # pyright: ignore[reportUnknownArgumentType]
                number_of_times_to_upsample,
                "cnn",
            )
        ]
    else:
        return [  # pyright: ignore[reportUnknownVariableType]
            _trim_css_to_bounds(
                _rect_to_css(
                    face,  # pyright: ignore[reportUnknownArgumentType]
                ),
                img.shape,  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
            )
            for face in _raw_face_locations(  # pyright: ignore[reportUnknownVariableType]
                img,  # pyright: ignore[reportUnknownArgumentType]
                number_of_times_to_upsample,
                model,
            )
        ]


def _raw_face_locations_batched(  # pyright: ignore[reportUnknownParameterType]
    images,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    number_of_times_to_upsample=1,  # pyright: ignore[reportMissingParameterType]
    batch_size=128,  # pyright: ignore[reportMissingParameterType]
):
    """
    Returns an 2d array of dlib rects of human faces in a image using the cnn face detector

    :param images: A list of images (each as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :return: A list of dlib 'rect' objects of found face locations
    """
    return cnn_face_detector(  # pyright: ignore[reportUnknownVariableType]
        images,
        number_of_times_to_upsample,
        batch_size=batch_size,
    )


def batch_face_locations(  # pyright: ignore[reportUnknownParameterType]
    images,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    number_of_times_to_upsample=1,  # pyright: ignore[reportMissingParameterType]
    batch_size=128,  # pyright: ignore[reportMissingParameterType]
):
    """
    Returns an 2d array of bounding boxes of human faces in a image using the cnn face detector
    If you are using a GPU, this can give you much faster results since the GPU
    can process batches of images at once. If you aren't using a GPU, you don't need this function.

    :param images: A list of images (each as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param batch_size: How many images to include in each GPU processing batch.
    :return: A list of tuples of found face locations in css (top, right, bottom, left) order
    """

    def convert_cnn_detections_to_css(  # pyright: ignore[reportUnknownParameterType]
        detections,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    ):
        return [  # pyright: ignore[reportUnknownVariableType]
            _trim_css_to_bounds(
                _rect_to_css(
                    face.rect,  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
                ),
                images[  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
                    0
                ].shape,
            )
            for face in detections  # pyright: ignore[reportUnknownVariableType]
        ]

    raw_detections_batched = (  # pyright: ignore[reportUnknownVariableType]
        _raw_face_locations_batched(
            images,  # pyright: ignore[reportUnknownArgumentType]
            number_of_times_to_upsample,
            batch_size,
        )
    )

    return list(  # pyright: ignore[reportUnknownVariableType]
        map(  # pyright: ignore[reportUnknownArgumentType]
            convert_cnn_detections_to_css,  # pyright: ignore[reportUnknownArgumentType]
            raw_detections_batched,  # pyright: ignore[reportUnknownArgumentType]
        ),
    )


def _raw_face_landmarks(  # pyright: ignore[reportUnknownParameterType]
    face_image,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    face_locations=None,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    model="large",  # pyright: ignore[reportMissingParameterType]
):
    if face_locations is None:
        face_locations = (  # pyright: ignore[reportUnknownVariableType]
            _raw_face_locations(
                face_image,  # pyright: ignore[reportUnknownArgumentType]
            )
        )
    else:
        face_locations = [  # pyright: ignore[reportUnknownVariableType]
            _css_to_rect(
                face_location,  # pyright: ignore[reportUnknownArgumentType]
            )
            for face_location in face_locations  # pyright: ignore[reportUnknownVariableType]
        ]

    pose_predictor = (  # pyright: ignore[reportUnknownVariableType]
        pose_predictor_68_point
    )

    if model == "small":
        pose_predictor = (  # pyright: ignore[reportUnknownVariableType]
            pose_predictor_5_point
        )

    return [  # pyright: ignore[reportUnknownVariableType]
        pose_predictor(
            face_image,
            face_location,
        )
        for face_location in face_locations  # pyright: ignore[reportUnknownVariableType]
    ]


def face_landmarks(  # pyright: ignore[reportUnknownParameterType]
    face_image,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    face_locations=None,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    model="large",  # pyright: ignore[reportMissingParameterType]
):
    """
    Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in the image

    :param face_image: image to search
    :param face_locations: Optionally provide a list of face locations to check.
    :param model: Optional - which model to use. "large" (default) or "small" which only returns 5 points but is faster.
    :return: A list of dicts of face feature locations (eyes, nose, etc)
    """
    landmarks = _raw_face_landmarks(  # pyright: ignore[reportUnknownVariableType]
        face_image,  # pyright: ignore[reportUnknownArgumentType]
        face_locations,
        model,
    )
    landmarks_as_tuples = [  # pyright: ignore[reportUnknownVariableType]
        [
            (
                p.x,  # pyright: ignore[reportUnknownMemberType]
                p.y,  # pyright: ignore[reportUnknownMemberType]
            )
            for p in landmark.parts()  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        ]
        for landmark in landmarks  # pyright: ignore[reportUnknownVariableType]
    ]

    # For a definition of each point index, see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
    if model == "large":
        return [  # pyright: ignore[reportUnknownVariableType]
            {
                "chin": points[0:17],
                "left_eyebrow": points[17:22],
                "right_eyebrow": points[22:27],
                "nose_bridge": points[27:31],
                "nose_tip": points[31:36],
                "left_eye": points[36:42],
                "right_eye": points[42:48],
                "top_lip": points[48:55]
                + [points[64]]
                + [points[63]]
                + [points[62]]
                + [points[61]]
                + [points[60]],
                "bottom_lip": points[54:60]
                + [points[48]]
                + [points[60]]
                + [points[67]]
                + [points[66]]
                + [points[65]]
                + [points[64]],
            }
            for points in landmarks_as_tuples  # pyright: ignore[reportUnknownVariableType]
        ]
    elif model == "small":
        return [  # pyright: ignore[reportUnknownVariableType]
            {
                "nose_tip": [points[4]],
                "left_eye": points[2:4],
                "right_eye": points[0:2],
            }
            for points in landmarks_as_tuples  # pyright: ignore[reportUnknownVariableType]
        ]
    else:
        raise ValueError(
            "Invalid landmarks model type. Supported models are ['small', 'large']."
        )


def face_encodings(  # pyright: ignore[reportUnknownParameterType]
    face_image,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    known_face_locations=None,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    num_jitters=1,  # pyright: ignore[reportMissingParameterType]
    model="small",  # pyright: ignore[reportMissingParameterType]
):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :param model: Optional - which model to use. "large" or "small" (default) which only returns 5 points but is faster.
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """
    raw_landmarks = _raw_face_landmarks(  # pyright: ignore[reportUnknownVariableType]
        face_image,  # pyright: ignore[reportUnknownArgumentType]
        known_face_locations,
        model,
    )
    return [  # pyright: ignore[reportUnknownVariableType]
        np.array(
            face_encoder.compute_face_descriptor(  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
                face_image,
                raw_landmark_set,
                num_jitters,
            ),
        )
        for raw_landmark_set in raw_landmarks  # pyright: ignore[reportUnknownVariableType]
    ]


def compare_faces(
    known_face_encodings,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    face_encoding_to_check,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    tolerance=0.6,  # pyright: ignore[reportMissingParameterType]
):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return list(
        face_distance(
            known_face_encodings,  # pyright: ignore[reportUnknownArgumentType]
            face_encoding_to_check,  # pyright: ignore[reportUnknownArgumentType]
        )
        <= tolerance,
    )
