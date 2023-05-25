# -*- coding: utf-8 -*-
from __future__ import print_function
import click
import os
import re
import face_recognition.api as face_recognition
import multiprocessing
import itertools
import sys
import PIL.Image
import numpy as np


def scan_known_people(  # pyright: ignore[reportUnknownParameterType]
    known_people_folder,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    known_names = []
    known_face_encodings = []

    for file in image_files_in_folder(  # pyright: ignore[reportUnknownVariableType]
        known_people_folder,  # pyright: ignore[reportUnknownArgumentType]
    ):
        basename = os.path.splitext(  # pyright: ignore[reportUnknownVariableType]
            os.path.basename(  # pyright: ignore[reportUnknownArgumentType]
                file,  # pyright: ignore[reportUnknownArgumentType]
            ),
        )[0]
        img = face_recognition.load_image_file(  # pyright: ignore[reportUnknownMemberType]
            file,  # pyright: ignore[reportUnknownArgumentType]
        )
        encodings = face_recognition.face_encodings(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            img,
        )

        if (
            len(
                encodings,  # pyright: ignore[reportUnknownArgumentType]
            )
            > 1
        ):
            click.echo(
                "WARNING: More than one face found in {}. Only considering the first face.".format(
                    file,  # pyright: ignore[reportUnknownArgumentType]
                ),
            )

        if (
            len(
                encodings,  # pyright: ignore[reportUnknownArgumentType]
            )
            == 0
        ):
            click.echo(
                "WARNING: No faces found in {}. Ignoring file.".format(
                    file,  # pyright: ignore[reportUnknownArgumentType]
                ),
            )
        else:
            known_names.append(  # pyright: ignore[reportUnknownMemberType]
                basename,  # pyright: ignore[reportUnknownArgumentType]
            )
            known_face_encodings.append(  # pyright: ignore[reportUnknownMemberType]
                encodings[0],
            )

    return (
        known_names,
        known_face_encodings,
    )  # pyright: ignore[reportUnknownVariableType]


def print_result(
    filename,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    name,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    distance,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    show_distance=False,  # pyright: ignore[reportMissingParameterType]
):
    if show_distance:
        print(
            "{},{},{}".format(
                filename,  # pyright: ignore[reportUnknownArgumentType]
                name,  # pyright: ignore[reportUnknownArgumentType]
                distance,  # pyright: ignore[reportUnknownArgumentType]
            ),
        )
    else:
        print(
            "{},{}".format(
                filename,  # pyright: ignore[reportUnknownArgumentType]
                name,  # pyright: ignore[reportUnknownArgumentType]
            ),
        )


def test_image(
    image_to_check,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    known_names,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    known_face_encodings,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    tolerance=0.6,  # pyright: ignore[reportMissingParameterType]
    show_distance=False,  # pyright: ignore[reportMissingParameterType]
):
    unknown_image = (
        face_recognition.load_image_file(  # pyright: ignore[reportUnknownMemberType]
            image_to_check,  # pyright: ignore[reportUnknownArgumentType]
        )
    )

    # Scale down image if it's giant so things run a little faster
    if max(unknown_image.shape) > 1600:
        pil_img = PIL.Image.fromarray(  # pyright: ignore[reportUnknownMemberType]
            unknown_image,
        )
        pil_img.thumbnail((1600, 1600), PIL.Image.LANCZOS)
        unknown_image = np.array(pil_img)

    unknown_encodings = face_recognition.face_encodings(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        unknown_image,
    )

    for (
        unknown_encoding
    ) in unknown_encodings:  # pyright: ignore[reportUnknownVariableType]
        distances = (
            face_recognition.face_distance(  # pyright: ignore[reportUnknownMemberType]
                known_face_encodings,  # pyright: ignore[reportUnknownArgumentType]
                unknown_encoding,
            )
        )
        result = list(distances <= tolerance)

        if True in result:
            [
                print_result(
                    image_to_check,  # pyright: ignore[reportUnknownArgumentType]
                    name,  # pyright: ignore[reportUnknownArgumentType]
                    distance,  # pyright: ignore[reportUnknownArgumentType]
                    show_distance,
                )
                for is_match, name, distance in zip(  # pyright: ignore[reportGeneralTypeIssues,reportUnknownVariableType]
                    result,
                    known_names,  # pyright: ignore[reportUnknownArgumentType]
                    distances,
                )
                if is_match
            ]
        else:
            print_result(
                image_to_check,  # pyright: ignore[reportUnknownArgumentType]
                "unknown_person",
                None,
                show_distance,
            )

    if not unknown_encodings:
        # print out fact that no faces were found in image
        print_result(
            image_to_check,  # pyright: ignore[reportUnknownArgumentType]
            "no_persons_found",
            None,
            show_distance,
        )


def image_files_in_folder(  # pyright: ignore[reportUnknownParameterType]
    folder,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    return [  # pyright: ignore[reportUnknownVariableType]
        os.path.join(
            folder,  # pyright: ignore[reportUnknownArgumentType]
            f,  # pyright: ignore[reportUnknownArgumentType]
        )
        for f in os.listdir(  # pyright: ignore[reportUnknownVariableType]
            folder,  # pyright: ignore[reportUnknownArgumentType]
        )
        if re.match(
            r".*\.(jpg|jpeg|png)",
            f,  # pyright: ignore[reportUnknownArgumentType]
            flags=re.I,
        )
    ]


def process_images_in_process_pool(
    images_to_check,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    known_names,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    known_face_encodings,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    number_of_cpus,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    tolerance,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    show_distance,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus  # pyright: ignore[reportUnknownVariableType]

    # macOS will crash due to a bug in libdispatch if you don't use 'forkserver'
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(
        processes=processes,  # pyright: ignore[reportUnknownArgumentType]
    )

    function_parameters = zip(
        images_to_check,  # pyright: ignore[reportUnknownArgumentType]
        itertools.repeat(  # pyright: ignore[reportUnknownArgumentType]
            known_names,  # pyright: ignore[reportUnknownArgumentType]
        ),
        itertools.repeat(  # pyright: ignore[reportUnknownArgumentType]
            known_face_encodings,  # pyright: ignore[reportUnknownArgumentType]
        ),
        itertools.repeat(  # pyright: ignore[reportUnknownArgumentType]
            tolerance,  # pyright: ignore[reportUnknownArgumentType]
        ),
        itertools.repeat(  # pyright: ignore[reportUnknownArgumentType]
            show_distance,  # pyright: ignore[reportUnknownArgumentType]
        ),
    )

    pool.starmap(
        test_image,  # pyright: ignore[reportUnknownArgumentType]
        function_parameters,
    )


@click.command()
@click.argument("known_people_folder")
@click.argument("image_to_check")
@click.option(
    "--cpus",
    default=1,
    help='number of CPU cores to use in parallel (can speed up processing lots of images). -1 means "use all in system"',
)
@click.option(
    "--tolerance",
    default=0.6,
    help="Tolerance for face comparisons. Default is 0.6. Lower this if you get multiple matches for the same person.",
)
@click.option(
    "--show-distance",
    default=False,
    type=bool,
    help="Output face distance. Useful for tweaking tolerance setting.",
)
def main(
    known_people_folder,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    image_to_check,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    cpus,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    tolerance,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    show_distance,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    (
        known_names,
        known_face_encodings,
    ) = scan_known_people(
        known_people_folder,  # pyright: ignore[reportUnknownArgumentType]
    )

    # Multi-core processing only supported on Python 3.4 or greater
    if (sys.version_info < (3, 4)) and cpus != 1:
        click.echo(
            "WARNING: Multi-processing support requires Python 3.4 or greater. Falling back to single-threaded processing!"
        )
        cpus = 1

    if os.path.isdir(
        image_to_check,  # pyright: ignore[reportUnknownArgumentType]
    ):
        if cpus == 1:
            [
                test_image(
                    image_file,  # pyright: ignore[reportUnknownArgumentType]
                    known_names,
                    known_face_encodings,
                    tolerance,  # pyright: ignore[reportUnknownArgumentType]
                    show_distance,  # pyright: ignore[reportUnknownArgumentType]
                )
                for image_file in image_files_in_folder(  # pyright: ignore[reportUnknownVariableType]
                    image_to_check,  # pyright: ignore[reportUnknownArgumentType]
                )
            ]
        else:
            process_images_in_process_pool(
                image_files_in_folder(
                    image_to_check,  # pyright: ignore[reportUnknownArgumentType]
                ),
                known_names,
                known_face_encodings,
                cpus,  # pyright: ignore[reportUnknownArgumentType]
                tolerance,  # pyright: ignore[reportUnknownArgumentType]
                show_distance,  # pyright: ignore[reportUnknownArgumentType]
            )
    else:
        test_image(
            image_to_check,  # pyright: ignore[reportUnknownArgumentType]
            known_names,
            known_face_encodings,
            tolerance,  # pyright: ignore[reportUnknownArgumentType]
            show_distance,  # pyright: ignore[reportUnknownArgumentType]
        )


if __name__ == "__main__":
    main()
