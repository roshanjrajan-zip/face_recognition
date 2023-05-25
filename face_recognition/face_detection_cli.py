# -*- coding: utf-8 -*-
from __future__ import print_function
import click
import os
import re
import face_recognition.api as face_recognition
import multiprocessing
import sys
import itertools


def print_result(
    filename,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    location,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    top, right, bottom, left = location  # pyright: ignore[reportUnknownVariableType]
    print(
        "{},{},{},{},{}".format(
            filename,  # pyright: ignore[reportUnknownArgumentType]
            top,  # pyright: ignore[reportUnknownArgumentType]
            right,  # pyright: ignore[reportUnknownArgumentType]
            bottom,  # pyright: ignore[reportUnknownArgumentType]
            left,  # pyright: ignore[reportUnknownArgumentType]
        ),
    )


def test_image(
    image_to_check,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    model,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    upsample,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    unknown_image = (
        face_recognition.load_image_file(  # pyright: ignore[reportUnknownMemberType]
            image_to_check,  # pyright: ignore[reportUnknownArgumentType]
        )
    )
    face_locations = face_recognition.face_locations(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        unknown_image,
        number_of_times_to_upsample=upsample,  # pyright: ignore[reportUnknownArgumentType]
        model=model,  # pyright: ignore[reportUnknownArgumentType]
    )

    for face_location in face_locations:  # pyright: ignore[reportUnknownVariableType]
        print_result(
            image_to_check,  # pyright: ignore[reportUnknownArgumentType]
            face_location,
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
    number_of_cpus,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    model,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    upsample,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
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
            model,  # pyright: ignore[reportUnknownArgumentType]
        ),
        itertools.repeat(  # pyright: ignore[reportUnknownArgumentType]
            upsample,  # pyright: ignore[reportUnknownArgumentType]
        ),
    )

    pool.starmap(
        test_image,  # pyright: ignore[reportUnknownArgumentType]
        function_parameters,
    )


@click.command()
@click.argument("image_to_check")
@click.option(
    "--cpus",
    default=1,
    help='number of CPU cores to use in parallel. -1 means "use all in system"',
)
@click.option(
    "--model",
    default="hog",
    help='Which face detection model to use. Options are "hog" or "cnn".',
)
@click.option(
    "--upsample",
    default=0,
    help="How many times to upsample the image looking for faces. Higher numbers find smaller faces.",
)
def main(
    image_to_check,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    cpus,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    model,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    upsample,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
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
                    model,  # pyright: ignore[reportUnknownArgumentType]
                    upsample,  # pyright: ignore[reportUnknownArgumentType]
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
                cpus,  # pyright: ignore[reportUnknownArgumentType]
                model,  # pyright: ignore[reportUnknownArgumentType]
                upsample,  # pyright: ignore[reportUnknownArgumentType]
            )
    else:
        test_image(
            image_to_check,  # pyright: ignore[reportUnknownArgumentType]
            model,  # pyright: ignore[reportUnknownArgumentType]
            upsample,  # pyright: ignore[reportUnknownArgumentType]
        )


if __name__ == "__main__":
    main()
