class CalibrationError(Exception):
    pass


class CameraDiffersException(CalibrationError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
