from transformers import AutoProcessor, Wav2Vec2Model

import numpy as np
import torch


def normalize(x, epsilon=1e-8):
    """
    Simple function that normalises the audio samples.
    """
    maxim, mini = x.max(), x.min()
    x_normed = (x - mini) / (
        (maxim - mini) + epsilon
    )  # epsilon to avoid dividing by zero
    return x_normed


def exctract_wav2vec(audios, sampling_rate=16000):
    """
    Wrapper for the wav2vec model, outputs only the hidden state (features).
    """

    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    features = []
    L = audios.shape[0]

    for i in range(L):
        print("sample {} / {}".format(i, L), end="\r")

        inputs = processor(
            audios[i, :], sampling_rate=sampling_rate, return_tensors="pt"
        )  # Batch size 1

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        features.append(last_hidden_states.detach().numpy())

    return features
