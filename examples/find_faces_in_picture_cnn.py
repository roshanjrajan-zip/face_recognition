from PIL import Image
import face_recognition

# Load the jpg file into a numpy array
image = face_recognition.load_image_file(  # pyright: ignore[reportUnknownMemberType]
    "biden.jpg",
)

# Find all the faces in the image using a pre-trained convolutional neural network.
# This method is more accurate than the default HOG model, but it's slower
# unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
# this will use GPU acceleration and perform well.
# See also: find_faces_in_picture.py
face_locations = face_recognition.face_locations(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    image,
    number_of_times_to_upsample=0,
    model="cnn",
)

print(
    "I found {} face(s) in this photograph.".format(
        len(
            face_locations,  # pyright: ignore[reportUnknownArgumentType]
        ),
    ),
)

for face_location in face_locations:  # pyright: ignore[reportUnknownVariableType]
    # Print the location of each face in this image
    (
        top,
        right,  # pyright: ignore[reportUnknownVariableType]
        bottom,  # pyright: ignore[reportUnknownVariableType]
        left,
    ) = face_location
    print(
        "A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(
            top,
            left,
            bottom,  # pyright: ignore[reportUnknownArgumentType]
            right,  # pyright: ignore[reportUnknownArgumentType]
        ),
    )

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(  # pyright: ignore[reportUnknownMemberType]
        face_image,
    )
    pil_image.show()
