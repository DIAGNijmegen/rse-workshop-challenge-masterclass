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
import torch
import monai
import json
from glob import glob
from pathlib import Path
from scipy.special import expit
from skimage import transform


def run():
    ## Read the inputs
    # OCT image
    input_oct_image = load_image_file_as_array(
        location= Path("/input") / "images" / "oct",
    )
    # dummy patient metadata that we will ignore
    # this is just to demonstrate the possibility of having multiple different inputs
    input_age_in_months = load_json_file(
         location=Path("/input") / "age-in-months.json",
    )

    # preprocess image
    original_shape = input_oct_image.shape
    preprocessed_oct_image = preprocess_image(image=input_oct_image)

    # load model and predict 
    result = predict(
        image=preprocessed_oct_image, 
    )

    # post-process image
    output_vessel_segmentation = postprocess_image(image=result, original_shape=original_shape)

    # Save your output
    write_array_as_image_file(
        location=Path("/output") / "images/vessel-segmentation",
        array=output_vessel_segmentation,
    )
    
    return 0


def preprocess_image(*, image):
    """Resize, normalize, and transpose the input image."""
    image = transform.resize(image, (512, 512), order=3) 
    image = image.astype(np.float32) / 255.  # normalize
    image = image.transpose((2, 0, 1))  # flip the axes and bring color to the first channel
    return image 


def postprocess_image(*, image, original_shape):
    """Resize the predicted image, binarize, and convert to uint8."""
    image = transform.resize(image, original_shape[:-1], order=3)
    image = (expit(image) > 0.99)  # apply the sigmoid filter and binarize the predictions
    image = (image * 255).astype(np.uint8) # cast to uint8, as expected for segmentations on Grand Challenge
    return image 


def predict(*, image):
    """Predict using a pre-trained UNet model."""
    device = torch.device('cpu')
    image = torch.from_numpy(image).to(device).reshape(1, 3, 512, 512)
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    model.eval()

    model.load_state_dict(
        torch.load(
            "model/model.pth",
            map_location=device,
        )
    )

    result = model(image).squeeze().data.cpu().numpy()

    return result


def load_json_file(*, location):
    # Reads a json file
    with open(location, 'r') as f:
        return json.loads(f.read())


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def write_array_as_image_file(*, location, array):
    location.mkdir(parents=True, exist_ok=True)
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


if __name__ == "__main__":
    raise SystemExit(run())
