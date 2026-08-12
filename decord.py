# stub: decord has no macOS arm64 wheels for py3.12; only needed for video input
class VideoReader:
    def __init__(self, *a, **kw):
        raise NotImplementedError("decord stub — video input not supported on this machine")

def cpu(*a, **kw): return None
def gpu(*a, **kw): return None

class bridge:
    @staticmethod
    def set_bridge(*a, **kw): pass