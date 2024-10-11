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
from pathlib import Path
import json
from glob import glob
import SimpleITK
import random

def run():
    # Read the input
    input_mammal = load_image_file_as_array(
        location= Path("/input") / "images" / "mammal",
    )
    input_age_in_months = load_json_file(
         location=Path("/input") / "age-in-months.json",
    )
    
    # For now, let us make bogus predictions
    output_uncertainty_score = 0.5
    output_cat_or_dog = random.choice(["feline", "canine"])

    # Save your output
    write_json_file(
        location=Path("/output")  / "uncertainty-score.json",
        content=output_uncertainty_score
    )
    write_json_file(
        location=Path("/output") / "cat_or_dog.json",
        content=output_cat_or_dog
    )
    
    return 0


def load_json_file(*, location):
    # Reads a json file
    with open(location, 'r') as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)

if __name__ == "__main__":
    raise SystemExit(run())
