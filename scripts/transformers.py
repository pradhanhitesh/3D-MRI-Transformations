import nibabel
import numpy as np
import matplotlib.pyplot as plt
import os

class MeanTransformation():
    """3D-MR Mean Transformation

    Parameters
    ------------
    filepath : str
    Define the filepath of .nii image

    axis : str, default = "coronal"
    Define the axis of the brain

    add_channel : bool, default = False
    get_std : bool, default = False
    """
    
    _parameter_constraints: dict = {
        "axis": ['coronal', 'axial', 'sagittal']
    }

    def __init__(
        self,
        filepath,
        axis="coronal",
        add_channel=False,
    ):
        if axis not in self._parameter_constraints["axis"]:
            raise ValueError(f"Invalid value for 'axis'. Allowed values are {self._parameter_constraints['axis']}")

        if not os.path.isfile(filepath):
            raise ValueError(f"Invalid file: {filepath}")

        self.axis = axis
        self.add_channel = add_channel
        self.filepath = filepath

    def _load_nii(self, filepath):
        image = nibabel.load(filepath)
        image_data = image.get_fdata()

        return image_data

    def _get_axis_number(self, axis):
        axis_info = {'sagittal': 0, 'axial': 1, 'coronal': 2}
        return axis_info[axis]

    def _get_mean_image(self, image):
        axis_number = self._get_axis_number(self.axis)
        mean_image = np.mean(image, axis=axis_number)
        if self.add_channel:
            norm_mean = (mean_image - np.min(mean_image)) / (np.max(mean_image) - np.min(mean_image))
            mean_image = np.stack([norm_mean, norm_mean, norm_mean], axis=-1)

        return mean_image

    def _get_std_image(self, image):
        axis_number = self._get_axis_number(self.axis)
        std_image = np.mean(image, axis=axis_number)
        if self.add_channel:
            norm_std = (std_image - np.min(std_image)) / (np.max(std_image) - np.min(std_image))
            std_image = np.stack([norm_std, norm_std, norm_std], axis=-1)

        return std_image

    def get_mean_image(self):
        image_data = self._load_nii(self.filepath)
        mean_image = self._get_mean_image(image_data)
        std_image = self._get_std_image(image_data)

        return mean_image, std_image