"""
The following is a simple example evaluation method.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the evaluation, reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""

import json
from glob import glob
import SimpleITK
import numpy as np
import random
from statistics import mean
from pathlib import Path
from pprint import pformat, pprint
from helpers import run_prediction_processing

INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
GROUND_TRUTH_DIRECTORY = Path("ground_truth")


def process(job):
    """Processes a single algorithm job, looking at the outputs"""
    report = "Processing:\n"
    report += pformat(job)
    report += "\n"

    # Firstly, find the location of the results
    location_retinal_vessel_segmentation = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="binary-vessel-segmentation",
    )

    # Secondly, read the results
    result_retinal_vessel_segmentation = load_image_file(
        location=location_retinal_vessel_segmentation,
    )

    # Thirdly, retrieve the input file name to match it with your ground truth
    image_name_oct_image = get_image_name(
        values=job["inputs"],
        slug="oct-image",
    )

    # Fourthly, load your ground truth
    matching_ground_truth_filename = {
        "11_oct_image.tif": "01_vessel_segmentation.mha",
        "12_oct_image.tif": "05_vessel_segmentation.mha",
        "13_oct_image.tif": "08_vessel_segmentation.mha",
    }[image_name_oct_image]

    ground_truth = get_array_from_image(
        image_path=GROUND_TRUTH_DIRECTORY / matching_ground_truth_filename
    )

    # Lastly, compare the results to your ground truth and compute some metrics

    # The Dice coefficient (also known as Sørensen–Dice index) is a statistical measure
    # used to gauge the similarity between two sets of data. In the context of binary
    # images or segmentation masks, it measures the overlap between two arrays.

    dice = dice_coefficient(ground_truth, result_retinal_vessel_segmentation)
    report += f"Dice: {dice}"

    print(report)

    # Finally, calculate by comparing the ground truth to the actual results
    return {
        "Dice": dice,
    }


def main():
    print_inputs()

    metrics = {}
    predictions = read_predictions()

    # We now process each algorithm job for this submission
    # Note that the jobs are not in any order!
    # We work that out from predictions.json

    # Use concurrent workers to process the predictions more efficiently
    metrics["results"] = run_prediction_processing(fn=process, predictions=predictions)

    # We have the results per prediction, we can aggregate over the results and
    # generate an overall score(s) for this submission
    metrics["aggregates"] = {
        "Dice": mean(result["Dice"] for result in metrics["results"])
    }

    # Make sure to save the metrics
    write_metrics(metrics=metrics)

    return 0


def print_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    input_files = [str(x) for x in Path(INPUT_DIRECTORY).rglob("*") if x.is_file()]

    print("Input Files:")
    pprint(input_files)
    print("")


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    with open(INPUT_DIRECTORY / "predictions.json") as f:
        return json.loads(f.read())


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_image_file(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )

    if len(input_files) != 1:
        raise RuntimeError(
            f"Could not load a single images from {location}. "
            "Ensure either .tif, .tiff or .mha files are present."
        )

    return get_array_from_image(image_path=input_files[0])


def get_array_from_image(image_path):
    result = SimpleITK.ReadImage(image_path)

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))


def dice_coefficient(array1, array2):
    # Ensure the arrays are binary (0 and 1 values)
    array1 = np.asarray(array1).astype(bool)
    array2 = np.asarray(array2).astype(bool)

    # Compute the intersection between the two arrays
    intersection = np.sum(array1 & array2)

    # Compute the Dice coefficient
    dice = (2.0 * intersection) / (np.sum(array1) + np.sum(array2))

    return dice


if __name__ == "__main__":
    raise SystemExit(main())
