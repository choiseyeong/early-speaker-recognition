from .xvector    import XVector
from .ecapa_tdnn import ECAPA_TDNN
from .resnet     import ResNet34SE

MODEL_REGISTRY = {
    'xvector'   : XVector,
    'ecapa_tdnn': ECAPA_TDNN,
    'resnet34se': ResNet34SE,
}

def build_model(name: str, num_speakers: int):
    """
    모델 이름으로 인스턴스 생성.

    Args:
        name        : 'xvector', 'ecapa_tdnn', 'resnet34se' 중 하나
        num_speakers: 전처리 후 실제 화자 수 (speaker_to_label.json 기준)
    """
    assert name in MODEL_REGISTRY, f"모델 '{name}'을 찾을 수 없습니다. 선택지: {list(MODEL_REGISTRY)}"
    return MODEL_REGISTRY[name](num_speakers=num_speakers)
