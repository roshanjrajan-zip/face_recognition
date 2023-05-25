import face_recognition
import cv2

# This code finds all faces in a list of images using the CNN model.
#
# This demo is for the _special case_ when you need to find faces in LOTS of images very quickly and all the images
# are the exact same size. This is common in video processing applications where you have lots of video frames
# to process.
#
# If you are processing a lot of images and using a GPU with CUDA, batch processing can be ~3x faster then processing
# single images at a time. But if you aren't using a GPU, then batch processing isn't going to be very helpful.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read the video file.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open video file
video_capture = cv2.VideoCapture(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    "short_hamilton_clip.mp4",
)

frames = []
frame_count = 0

while video_capture.isOpened():  # pyright: ignore[reportUnknownMemberType]
    # Grab a single frame of video
    (
        ret,  # pyright: ignore[reportUnknownVariableType]
        frame,  # pyright: ignore[reportUnknownVariableType]
    ) = video_capture.read()  # pyright: ignore[reportUnknownMemberType]

    # Bail out when the video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    frame = frame[:, :, ::-1]  # pyright: ignore[reportUnknownVariableType]

    # Save each frame of the video to a list
    frame_count += 1
    frames.append(  # pyright: ignore[reportUnknownMemberType]
        frame,  # pyright: ignore[reportUnknownArgumentType]
    )

    # Every 128 frames (the default batch size), batch process the list of frames to find faces
    if len(frames) == 128:
        batch_of_face_locations = face_recognition.batch_face_locations(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            frames,
            number_of_times_to_upsample=0,
        )

        # Now let's list all the faces we found in all 128 frames
        for (
            frame_number_in_batch,
            face_locations,  # pyright: ignore[reportUnknownVariableType]
        ) in enumerate(
            batch_of_face_locations,  # pyright: ignore[reportUnknownArgumentType]
        ):
            number_of_faces_in_frame = len(
                face_locations,  # pyright: ignore[reportUnknownArgumentType]
            )

            frame_number = frame_count - 128 + frame_number_in_batch
            print(
                "I found {} face(s) in frame #{}.".format(
                    number_of_faces_in_frame, frame_number
                )
            )

            for (
                face_location
            ) in face_locations:  # pyright: ignore[reportUnknownVariableType]
                # Print the location of each face in this frame
                (
                    top,
                    right,  # pyright: ignore[reportUnknownVariableType]
                    bottom,  # pyright: ignore[reportUnknownVariableType]
                    left,
                ) = face_location
                print(
                    " - A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(
                        top,
                        left,
                        bottom,  # pyright: ignore[reportUnknownArgumentType]
                        right,  # pyright: ignore[reportUnknownArgumentType]
                    ),
                )

        # Clear the frames array to start the next batch
        frames = []
