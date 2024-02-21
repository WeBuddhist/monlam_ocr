class ImageReadError(Exception):
    """Exception raised when an image cannot be read or processed correctly."""

    def __init__(self, message="Image cannot be read or is empty"):
        self.message = message
        super().__init__(self.message)


class InvalidDimensionError(Exception):
    """Exception raised for errors in array dimensions."""

    def __init__(self, message="Invalid array dimension access"):
        self.message = message
        super().__init__(self.message)


class FailedToLoadModel(Exception):
    """Exception raised when a model cannot be loaded correctly."""

    def __init__(self, message="Model cannot be loaded"):
        self.message = message
        super().__init__(self.message)


class FailedToDownloadModel(Exception):
    """Exception raised when a model cannot be downloaded correctly."""

    def __init__(self, message="Model cannot be downloaded"):
        self.message = message
        super().__init__(self.message)
