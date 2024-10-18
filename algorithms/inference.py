"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""

import numpy as np
import json
import SimpleITK
from glob import glob
from uuid import uuid4
from pathlib import Path


THRESHOLD = 128

def run():
    ## Read the inputs
<<<<<<< HEAD
    # Fundus image
    input_fundus_image = load_image(
        location= Path("/input") / "images" / "color-fundus",
=======
    # OCT image
    input_color-fundus-image = load_image(
        location= Path("/input") / "images" / "oct",
>>>>>>> 7eb80256ee31b6e6cd5d3f7879785a5c423410eb
    )
    # Dummy patient metadata which we will ignore
    # This is just to demonstrate the possibility of having multiple different inputs
    input_age_in_months = load_json_file(
        location=Path("/input") / "age-in-months.json",
    )

    # Process inputs and generate predictions:
<<<<<<< HEAD
    # For this example, we will simply convert the image to a binary mask
    output_vessel_segmentation = convert_to_binary_mask(image=input_fundus_image)
=======
    # For this example, we will simply convert the image 
    # to a binary mask by applying some thresholding
    output_vessel_segmentation = convert_to_binary_mask(image=input_color-fundus-image)
>>>>>>> 7eb80256ee31b6e6cd5d3f7879785a5c423410eb
    
    # Save your output
    write_image_to_file(
        location=Path("/output") / "images/binary-vessel-segmentation",
        image=output_vessel_segmentation,
    )

    return 0


def convert_to_binary_mask(*, image):
    # Convert the image to grayscale by averaging the RGB channels
    gray_image = SimpleITK.VectorIndexSelectionCast(image, 0)
    for i in range(1, image.GetNumberOfComponentsPerPixel()):
        gray_image += SimpleITK.VectorIndexSelectionCast(image, i)
    gray_image /= image.GetNumberOfComponentsPerPixel()

    # Apply thresholding to binarize the image 
    # The resulting image's voxel values need to match those defined for the output interface on Grand Challenge 
    binary_mask = SimpleITK.BinaryThreshold(
        gray_image, 
        lowerThreshold=THRESHOLD, # lower bound of the pixel intensity range that will be considered "inside" the threshold range
        upperThreshold=255, # upper bound of the pixel intensity range that will be considered "inside" the threshold range
        insideValue=255, # value to assign to pixels that fall within the threshold range
        outsideValue=0 # value to assign to pixels that fall outside the threshold range
    )
    
    return binary_mask


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def load_image(*, location):
    # Use SimpleITK to read a file
    # The specified image folder will only contain 1 image 
    # because an algorithm only gets 1 archive item to process at a time
    input_files = glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])
    return result


def write_image_to_file(*, location, image):
    location.mkdir(parents=True, exist_ok=True)
    # Note that it doesn't matter what filename you give the output image
    # Grand Challenge will save the output under its own random uuid
    # Here, we generate a random unique identifier
    uuid = uuid4()
    # Grand Challenge expects image outputs to be either in MHA or TIFF format 
    # Here, we write the image as MHA file
    SimpleITK.WriteImage(
        image,
        location / f"{uuid}.mha",
        useCompression=True,
    )


if __name__ == "__main__":
    raise SystemExit(run())
