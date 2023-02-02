import glob
import os

import torch

import whisper
from whisper.decoding import DecodingOptions
import requests
import json

option = {
    "task": "transcribe",
    "verbose": True,
    "language": "en",
    "temperature": 0.0,
    "initial_prompt": [],
    "best_of": None,
    "beam_size": 3,
    "patience": 1.3,
    "length_penalty": None,
    "suppress_tokens": "-1",
    "without_timestamps": False,  #
    "condition_on_previous_text": False,
    "fp16": True,
    "no_speech_threshold": 0.9,
    "compression_ratio_threshold": None,  # 2.4,
    "logprob_threshold": None,
}

model = whisper.load_model("base")
# file = "jfk.flac"
file = "1.wav"
# bug_result = model.transcribe(bug_audio, **option)
# result = model.transcribe(file)
result = model.transcribe(file, **option)
print(result)
# print(result["text"])
