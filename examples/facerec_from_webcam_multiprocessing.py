import face_recognition
import cv2
from multiprocessing import Process, Manager, cpu_count, set_start_method
import time
import numpy
import threading
import platform


# This is a little bit complicated (but fast) example of running face recognition on live video from your webcam.
# This example is using multiprocess.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.


# Get next worker's id
def next_id(  # pyright: ignore[reportUnknownParameterType]
    current_id,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    worker_num,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    if current_id == worker_num:
        return 1
    else:
        return current_id + 1  # pyright: ignore[reportUnknownVariableType]


# Get previous worker's id
def prev_id(  # pyright: ignore[reportUnknownParameterType]
    current_id,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    worker_num,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    if current_id == 1:
        return worker_num  # pyright: ignore[reportUnknownVariableType]
    else:
        return current_id - 1  # pyright: ignore[reportUnknownVariableType]


# A subprocess use to capture frames.
def capture(
    read_frame_list,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    Global,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    worker_num,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        0,
    )
    # video_capture.set(3, 640)  # Width of the frames in the video stream.
    # video_capture.set(4, 480)  # Height of the frames in the video stream.
    # video_capture.set(5, 30) # Frame rate.
    print(
        "Width: %d, Height: %d, FPS: %d"
        % (
            video_capture.get(  # pyright: ignore[reportUnknownMemberType]
                3,
            ),
            video_capture.get(  # pyright: ignore[reportUnknownMemberType]
                4,
            ),
            video_capture.get(  # pyright: ignore[reportUnknownMemberType]
                5,
            ),
        ),
    )

    while not Global.is_exit:  # pyright: ignore[reportUnknownMemberType]
        # If it's time to read a frame
        if Global.buff_num != next_id(  # pyright: ignore[reportUnknownMemberType]
            Global.read_num,  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
            worker_num,  # pyright: ignore[reportUnknownArgumentType]
        ):
            # Grab a single frame of video
            (
                ret,  # pyright: ignore[reportUnknownVariableType]
                frame,  # pyright: ignore[reportUnknownVariableType]
            ) = video_capture.read()  # pyright: ignore[reportUnknownMemberType]
            read_frame_list[
                Global.buff_num  # pyright: ignore[reportUnknownMemberType]
            ] = frame
            Global.buff_num = next_id(
                Global.buff_num,  # pyright: ignore[reportUnknownMemberType]
                worker_num,  # pyright: ignore[reportUnknownArgumentType]
            )
        else:
            time.sleep(0.01)

    # Release webcam
    video_capture.release()  # pyright: ignore[reportUnknownMemberType]  # pyright: ignore[reportUnknownMemberType]


# Many subprocess use to process frames.
def process(
    worker_id,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    read_frame_list,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    write_frame_list,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    Global,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    worker_num,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    known_face_encodings = (  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        Global.known_face_encodings
    )
    known_face_names = (  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        Global.known_face_names
    )
    while not Global.is_exit:  # pyright: ignore[reportUnknownMemberType]
        # Wait to read
        while (
            Global.read_num != worker_id  # pyright: ignore[reportUnknownMemberType]
            or Global.read_num  # pyright: ignore[reportUnknownMemberType]
            != prev_id(
                Global.buff_num,  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
                worker_num,  # pyright: ignore[reportUnknownArgumentType]
            )
        ):
            # If the user has requested to end the app, then stop waiting for webcam frames
            if Global.is_exit:  # pyright: ignore[reportUnknownMemberType]
                break

            time.sleep(0.01)

        # Delay to make the video look smoother
        time.sleep(
            Global.frame_delay,  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
        )

        # Read a single frame from frame list
        frame_process = read_frame_list[  # pyright: ignore[reportUnknownVariableType]
            worker_id
        ]

        # Expect next worker to read frame
        Global.read_num = next_id(
            Global.read_num,  # pyright: ignore[reportUnknownMemberType]
            worker_num,  # pyright: ignore[reportUnknownArgumentType]
        )

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame_process[  # pyright: ignore[reportUnknownVariableType]
            :, :, ::-1
        ]

        # Find all the faces and face encodings in the frame of video, cost most time
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
            matches = face_recognition.compare_faces(  # pyright: ignore[reportUnknownMemberType]
                known_face_encodings,  # pyright: ignore[reportUnknownArgumentType]
                face_encoding,  # pyright: ignore[reportUnknownArgumentType]
            )

            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[  # pyright: ignore[reportUnknownVariableType]
                    first_match_index
                ]

            # Draw a box around the face
            cv2.rectangle(  # pyright: ignore[reportUnknownMemberType]
                frame_process,
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
                frame_process,
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
                frame_process,
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

        # Wait to write
        while Global.write_num != worker_id:  # pyright: ignore[reportUnknownMemberType]
            time.sleep(0.01)

        # Send frame to global
        write_frame_list[worker_id] = frame_process

        # Expect next worker to write frame
        Global.write_num = next_id(
            Global.write_num,  # pyright: ignore[reportUnknownMemberType]
            worker_num,  # pyright: ignore[reportUnknownArgumentType]
        )


if __name__ == "__main__":
    # Fix Bug on MacOS
    if platform.system() == "Darwin":
        set_start_method("forkserver")

    # Global variables
    Global = Manager().Namespace()
    Global.buff_num = 1
    Global.read_num = 1
    Global.write_num = 1
    Global.frame_delay = 0
    Global.is_exit = False
    read_frame_list = Manager().dict()
    write_frame_list = Manager().dict()

    # Number of workers (subprocess use to process frames)
    if cpu_count() > 2:
        worker_num = cpu_count() - 1  # 1 for capturing frames
    else:
        worker_num = 2

    # Subprocess list
    p = []

    # Create a thread to capture frames (if uses subprocess, it will crash on Mac)
    p.append(  # pyright: ignore[reportUnknownMemberType]
        threading.Thread(
            target=capture,  # pyright: ignore[reportUnknownArgumentType]
            args=(
                read_frame_list,
                Global,
                worker_num,
            ),
        ),
    )
    p[0].start()  # pyright: ignore[reportUnknownMemberType]

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
    Global.known_face_encodings = [obama_face_encoding, biden_face_encoding]
    Global.known_face_names = ["Barack Obama", "Joe Biden"]

    # Create workers
    for worker_id in range(1, worker_num + 1):
        p.append(  # pyright: ignore[reportUnknownMemberType]
            Process(
                target=process,  # pyright: ignore[reportUnknownArgumentType]
                args=(
                    worker_id,
                    read_frame_list,
                    write_frame_list,
                    Global,
                    worker_num,
                ),
            ),
        )
        p[worker_id].start()  # pyright: ignore[reportUnknownMemberType]

    # Start to show video
    last_num = 1
    fps_list = []
    tmp_time = time.time()
    while not Global.is_exit:
        while Global.write_num != last_num:
            last_num = int(Global.write_num)

            # Calculate fps
            delay = time.time() - tmp_time
            tmp_time = time.time()
            fps_list.append(  # pyright: ignore[reportUnknownMemberType]
                delay,
            )
            if len(fps_list) > 5 * worker_num:
                fps_list.pop(0)
            fps = len(  # pyright: ignore[reportUnknownVariableType]
                fps_list,
            ) / numpy.sum(  # pyright: ignore[reportUnknownMemberType]
                fps_list,
            )
            print(
                "fps: %.2f" % fps,  # pyright: ignore[reportUnknownArgumentType]
            )

            # Calculate frame delay, in order to make the video look smoother.
            # When fps is higher, should use a smaller ratio, or fps will be limited in a lower value.
            # Larger ratio can make the video look smoother, but fps will hard to become higher.
            # Smaller ratio can make fps higher, but the video looks not too smoother.
            # The ratios below are tested many times.
            if fps < 6:
                Global.frame_delay = (1 / fps) * 0.75
            elif fps < 20:
                Global.frame_delay = (1 / fps) * 0.5
            elif fps < 30:
                Global.frame_delay = (1 / fps) * 0.25
            else:
                Global.frame_delay = 0

            # Display the resulting image
            cv2.imshow(  # pyright: ignore[reportUnknownMemberType]
                "Video",
                write_frame_list[
                    prev_id(
                        Global.write_num,
                        worker_num,
                    )
                ],
            )

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(  # pyright: ignore[reportUnknownMemberType]
            1,
        ) & 0xFF == ord(
            "q",
        ):
            Global.is_exit = True
            break

        time.sleep(0.01)

    # Quit
    cv2.destroyAllWindows()  # pyright: ignore[reportUnknownMemberType]  # pyright: ignore[reportUnknownMemberType]
