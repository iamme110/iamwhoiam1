import abc
class PytorchVideoReader(abc.ABC):
    @abc.abstractmethod
    def __init__(self, file, device):
        pass

    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abc.abstractmethod
    def frames(self):
        pass

    @abc.abstractmethod
    def seek(self, offset_sec):
        pass


class PytorchVideoWriter(abc.ABC):
    @abc.abstractmethod
    def __init__(self, file, width, height, fps, codec, device, crf=None, preset=None, time_base=None, moov_front=False, custom_encoder_options=None):
        pass

    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abc.abstractmethod
    def write(self, frame, frame_pts=None, bgr2rgb=False):
        pass

    @abc.abstractmethod
    def get_default_encoder_options(self):
        pass

    @staticmethod
    def parse_custom_options(custom_encoder_options):
        # squeeze spaces
        custom_encoder_options = ' '.join(custom_encoder_options.split())
        import re
        regex = re.compile(r"-(\w+ \w+)")
        matches = regex.findall(custom_encoder_options)
        encoder_options = {}
        for match in matches:
            option, value = match.split()
            encoder_options[option] = value
        return encoder_options
