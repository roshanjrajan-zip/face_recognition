import face_recognition
from PIL import Image, ImageDraw
import numpy as np

# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.

# Load a sample picture and learn how to recognize it.
obama_image = (
    face_recognition.load_image_file(  # pyright: ignore[reportUnknownMemberType]
        "obama.jpg",
    )
)
obama_face_encoding = face_recognition.face_encodings(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    obama_image,
)[
    0
]

# Load a second sample picture and learn how to recognize it.
biden_image = (
    face_recognition.load_image_file(  # pyright: ignore[reportUnknownMemberType]
        "biden.jpg",
    )
)
biden_face_encoding = face_recognition.face_encodings(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    biden_image,
)[
    0
]

# Create arrays of known face encodings and their names
known_face_encodings = [  # pyright: ignore[reportUnknownVariableType]
    obama_face_encoding,
    biden_face_encoding,
]
known_face_names = ["Barack Obama", "Joe Biden"]

# Load an image with an unknown face
unknown_image = (
    face_recognition.load_image_file(  # pyright: ignore[reportUnknownMemberType]
        "two_people.jpg",
    )
)

# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    unknown_image,
)
face_encodings = face_recognition.face_encodings(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    unknown_image,
    face_locations,
)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
# See http://pillow.readthedocs.io/ for more about PIL/Pillow
pil_image = Image.fromarray(  # pyright: ignore[reportUnknownMemberType]
    unknown_image,
)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (  # pyright: ignore[reportGeneralTypeIssues]
    top,  # pyright: ignore[reportUnknownVariableType]
    right,  # pyright: ignore[reportUnknownVariableType]
    bottom,  # pyright: ignore[reportUnknownVariableType]
    left,  # pyright: ignore[reportUnknownVariableType]
), face_encoding in zip(  # pyright: ignore[reportUnknownVariableType]
    face_locations,  # pyright: ignore[reportUnknownArgumentType]
    face_encodings,  # pyright: ignore[reportUnknownArgumentType]
):
    # See if the face is a match for the known face(s)
    matches = (
        face_recognition.compare_faces(  # pyright: ignore[reportUnknownMemberType]
            known_face_encodings,
            face_encoding,  # pyright: ignore[reportUnknownArgumentType]
        )
    )

    name = "Unknown"

    # If a match was found in known_face_encodings, just use the first one.
    # if True in matches:
    #     first_match_index = matches.index(True)
    #     name = known_face_names[first_match_index]

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = (
        face_recognition.face_distance(  # pyright: ignore[reportUnknownMemberType]
            known_face_encodings,
            face_encoding,  # pyright: ignore[reportUnknownArgumentType]
        )
    )
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    # Draw a box around the face using the Pillow module
    draw.rectangle(
        (  # pyright: ignore[reportUnknownArgumentType]
            (left, top),
            (right, bottom),
        ),
        outline=(
            0,
            0,
            255,
        ),
    )

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(
        (  # pyright: ignore[reportUnknownArgumentType]
            (
                left,
                bottom - text_height - 10,
            ),
            (right, bottom),
        ),
        fill=(
            0,
            0,
            255,
        ),
        outline=(
            0,
            0,
            255,
        ),
    )
    draw.text(  # pyright: ignore[reportUnknownMemberType]
        (  # pyright: ignore[reportUnknownArgumentType]
            left + 6,
            bottom - text_height - 5,
        ),
        name,
        fill=(
            255,
            255,
            255,
            255,
        ),
    )


# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
pil_image.show()

# You can also save a copy of the new image to disk if you want by uncommenting this line
# pil_image.save("image_with_boxes.jpg")
