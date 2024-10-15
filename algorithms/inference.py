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
import SimpleITK
import numpy as np
import json
from glob import glob
from pathlib import Path


THRESHOLD = 128

def run():
    ## Read the inputs
    # OCT image
    input_oct_image = load_image(
        location= Path("/input") / "images" / "oct",
    )
    # dummy patient metadata that we will ignore
    # this is just to demonstrate the possibility of having multiple different inputs
    input_age_in_months = load_json_file(
         location=Path("/input") / "age-in-months.json",
    )

    # Process the image, generate predictions
    # For this example, we will simply convert the image 
    # to a binary mask by applying some thresholding
    output_vessel_segmentation = convert_to_binary_mask(image=input_oct_image)
    
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

    # Apply thresholding, return image with voxel values 
    # as defined for the output interface on Grand Challenge 
    binary_mask = SimpleITK.BinaryThreshold(
        gray_image, 
        lowerThreshold=THRESHOLD, 
        upperThreshold=255, 
        insideValue=255, 
        outsideValue=0
    )
    
    return binary_mask


def load_json_file(*, location):
    # Reads a json file
    with open(location, 'r') as f:
        return json.loads(f.read())


def load_image(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tif")) + glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])
    return result


def write_image_to_file(*, location, image):
    location.mkdir(parents=True, exist_ok=True)
    suffix = ".mha"
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


if __name__ == "__main__":
    raise SystemExit(run())
