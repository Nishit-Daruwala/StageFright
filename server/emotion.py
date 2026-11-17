"""Speech emotion recognition on short windows (wav2vec2 classifier)."""

import numpy as np

from . import config

LABEL_NAMES = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad", "fea": "fearful",
               "dis": "disgust", "sur": "surprised", "cal": "calm"}


class EmotionModel:
    def __init__(self):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        self._torch = torch
        torch.set_num_threads(2)
        local = config.EMOTION_LOCAL_DIR
        self.source = str(local) if (local / "config.json").exists() else config.EMOTION_FALLBACK
        self.extractor = AutoFeatureExtractor.from_pretrained(self.source)
        self.model = AutoModelForAudioClassification.from_pretrained(self.source).eval()
        self.labels = [LABEL_NAMES.get(v.lower()[:3], v.lower()) for _, v in sorted(self.model.config.id2label.items())]

    def predict(self, window: np.ndarray) -> dict:
        """Emotion of one audio window: {label, probs}."""
        inputs = self.extractor(window, sampling_rate=config.SR, return_tensors="pt")
        with self._torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probs = self._torch.softmax(logits, dim=-1).numpy()
        return {"label": self.labels[int(probs.argmax())],
                "probs": {l: round(float(p), 3) for l, p in zip(self.labels, probs)}}
