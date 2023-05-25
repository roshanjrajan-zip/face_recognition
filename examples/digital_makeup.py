from PIL import Image, ImageDraw
import face_recognition

# Load the jpg file into a numpy array
image = face_recognition.load_image_file(  # pyright: ignore[reportUnknownMemberType]
    "biden.jpg",
)

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    image,
)

pil_image = Image.fromarray(  # pyright: ignore[reportUnknownMemberType]
    image,
)
for face_landmarks in face_landmarks_list:  # pyright: ignore[reportUnknownVariableType]
    d = ImageDraw.Draw(pil_image, "RGBA")

    # Make the eyebrows into a nightmare
    d.polygon(
        face_landmarks["left_eyebrow"],  # pyright: ignore[reportUnknownArgumentType]
        fill=(
            68,
            54,
            39,
            128,
        ),
    )
    d.polygon(
        face_landmarks["right_eyebrow"],  # pyright: ignore[reportUnknownArgumentType]
        fill=(
            68,
            54,
            39,
            128,
        ),
    )
    d.line(
        face_landmarks["left_eyebrow"],  # pyright: ignore[reportUnknownArgumentType]
        fill=(
            68,
            54,
            39,
            150,
        ),
        width=5,
    )
    d.line(
        face_landmarks["right_eyebrow"],  # pyright: ignore[reportUnknownArgumentType]
        fill=(
            68,
            54,
            39,
            150,
        ),
        width=5,
    )

    # Gloss the lips
    d.polygon(
        face_landmarks["top_lip"],  # pyright: ignore[reportUnknownArgumentType]
        fill=(
            150,
            0,
            0,
            128,
        ),
    )
    d.polygon(
        face_landmarks["bottom_lip"],  # pyright: ignore[reportUnknownArgumentType]
        fill=(
            150,
            0,
            0,
            128,
        ),
    )
    d.line(
        face_landmarks["top_lip"],  # pyright: ignore[reportUnknownArgumentType]
        fill=(
            150,
            0,
            0,
            64,
        ),
        width=8,
    )
    d.line(
        face_landmarks["bottom_lip"],  # pyright: ignore[reportUnknownArgumentType]
        fill=(
            150,
            0,
            0,
            64,
        ),
        width=8,
    )

    # Sparkle the eyes
    d.polygon(
        face_landmarks["left_eye"],  # pyright: ignore[reportUnknownArgumentType]
        fill=(
            255,
            255,
            255,
            30,
        ),
    )
    d.polygon(
        face_landmarks["right_eye"],  # pyright: ignore[reportUnknownArgumentType]
        fill=(
            255,
            255,
            255,
            30,
        ),
    )

    # Apply some eyeliner
    d.line(
        face_landmarks["left_eye"]  # pyright: ignore[reportUnknownArgumentType]
        + [face_landmarks["left_eye"][0]],
        fill=(
            0,
            0,
            0,
            110,
        ),
        width=6,
    )
    d.line(
        face_landmarks["right_eye"]  # pyright: ignore[reportUnknownArgumentType]
        + [face_landmarks["right_eye"][0]],
        fill=(
            0,
            0,
            0,
            110,
        ),
        width=6,
    )

    pil_image.show()
