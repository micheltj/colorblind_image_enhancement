import numpy as np


class Transformer:
    """Interface for image/video transformers."""
    def transform(self) -> (np.array, np.array):
        """Generator over the provided images that returns a tuple of (the original image, the transformed image)."""
        pass

    def transform_single(self, image: np.array):
        """Transform a single image provided by parameter."""
        pass

    @property
    def provides_type(self) -> str:
        """Define the image format returned by this transformer (supported are bgr888, rgb888, gray8)."""
        # Note: Depending on this value automatic conversion takes place in ProgressionWorker/-Window to write and
        # display the video correctly
        return 'bgr888'
