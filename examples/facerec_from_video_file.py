import face_recognition
import cv2

# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
input_movie = cv2.VideoCapture(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    "hamilton_clip.mp4",
)
length = int(
    input_movie.get(  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
        cv2.CAP_PROP_FRAME_COUNT,  # pyright: ignore[reportUnknownMemberType]
    ),
)

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    *"XVID",
)
output_movie = cv2.VideoWriter(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    "output.avi",
    fourcc,
    29.97,
    (
        640,
        360,
    ),
)

# Load some sample pictures and learn how to recognize them.
lmm_image = (
    face_recognition.load_image_file(  # pyright: ignore[reportUnknownMemberType]
        "lin-manuel-miranda.png",
    )
)
lmm_face_encoding = face_recognition.face_encodings(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    lmm_image,
)[
    0
]

al_image = face_recognition.load_image_file(  # pyright: ignore[reportUnknownMemberType]
    "alex-lacamoire.png",
)
al_face_encoding = face_recognition.face_encodings(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    al_image,
)[
    0
]

known_faces = [  # pyright: ignore[reportUnknownVariableType]
    lmm_face_encoding,
    al_face_encoding,
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    (
        ret,  # pyright: ignore[reportUnknownVariableType]
        frame,  # pyright: ignore[reportUnknownVariableType]
    ) = input_movie.read()  # pyright: ignore[reportUnknownMemberType]
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]  # pyright: ignore[reportUnknownVariableType]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        rgb_frame,  # pyright: ignore[reportUnknownArgumentType]
    )
    face_encodings = face_recognition.face_encodings(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        rgb_frame,  # pyright: ignore[reportUnknownArgumentType]
        face_locations,
    )

    face_names = []
    for face_encoding in face_encodings:  # pyright: ignore[reportUnknownVariableType]
        # See if the face is a match for the known face(s)
        match = (
            face_recognition.compare_faces(  # pyright: ignore[reportUnknownMemberType]
                known_faces,
                face_encoding,
                tolerance=0.50,
            )
        )

        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        name = None
        if match[0]:
            name = "Lin-Manuel Miranda"
        elif match[1]:
            name = "Alex Lacamoire"

        face_names.append(  # pyright: ignore[reportUnknownMemberType]
            name,
        )

    # Label the results
    for (  # pyright: ignore[reportGeneralTypeIssues]
        top,  # pyright: ignore[reportUnknownVariableType]
        right,  # pyright: ignore[reportUnknownVariableType]
        bottom,  # pyright: ignore[reportUnknownVariableType]
        left,  # pyright: ignore[reportUnknownVariableType]
    ), name in zip(  # pyright: ignore[reportUnknownVariableType]
        face_locations,  # pyright: ignore[reportUnknownArgumentType]
        face_names,
    ):
        if not name:
            continue

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
                bottom - 25,
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
            0.5,
            (
                255,
                255,
                255,
            ),
            1,
        )

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(  # pyright: ignore[reportUnknownMemberType]
        frame,
    )

# All done!
input_movie.release()  # pyright: ignore[reportUnknownMemberType]  # pyright: ignore[reportUnknownMemberType]
cv2.destroyAllWindows()  # pyright: ignore[reportUnknownMemberType]  # pyright: ignore[reportUnknownMemberType]
