import face_recognition
import cv2
import numpy as np

# This is a super simple (but slow) example of running face recognition on live video from your webcam.
# There's a second example that's a little more complicated but runs faster.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    0,
)

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

while True:
    # Grab a single frame of video
    (
        ret,  # pyright: ignore[reportUnknownVariableType]
        frame,  # pyright: ignore[reportUnknownVariableType]
    ) = video_capture.read()  # pyright: ignore[reportUnknownMemberType]

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]  # pyright: ignore[reportUnknownVariableType]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        rgb_frame,  # pyright: ignore[reportUnknownArgumentType]
    )
    face_encodings = face_recognition.face_encodings(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        rgb_frame,  # pyright: ignore[reportUnknownArgumentType]
        face_locations,
    )

    # Loop through each face in this frame of video
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

        # Draw a box around the face
        cv2.rectangle(  # pyright: ignore[reportUnknownMemberType]
            frame,
            (
                left,
                top,
            ),
            (
                right,
                bottom,
            ),
            (
                0,
                0,
                255,
            ),
            2,
        )

        # Draw a label with a name below the face
        cv2.rectangle(  # pyright: ignore[reportUnknownMemberType]
            frame,
            (
                left,
                bottom - 35,
            ),
            (
                right,
                bottom,
            ),
            (
                0,
                0,
                255,
            ),
            cv2.FILLED,  # pyright: ignore[reportUnknownMemberType]
        )
        font = (  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            cv2.FONT_HERSHEY_DUPLEX
        )
        cv2.putText(  # pyright: ignore[reportUnknownMemberType]
            frame,
            name,
            (
                left + 6,
                bottom - 6,
            ),
            font,
            1.0,
            (
                255,
                255,
                255,
            ),
            1,
        )

    # Display the resulting image
    cv2.imshow(  # pyright: ignore[reportUnknownMemberType]
        "Video",
        frame,
    )

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(  # pyright: ignore[reportUnknownMemberType]
        1,
    ) & 0xFF == ord(
        "q",
    ):
        break

# Release handle to the webcam
video_capture.release()  # pyright: ignore[reportUnknownMemberType]  # pyright: ignore[reportUnknownMemberType]
cv2.destroyAllWindows()  # pyright: ignore[reportUnknownMemberType]  # pyright: ignore[reportUnknownMemberType]
