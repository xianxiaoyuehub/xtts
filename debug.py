import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2",
        "/workspace/pretrained/tts/tts_models--multilingual--multi-dataset--xtts_v2-520k",
        gpu='cuda')

his_audio = ['/workspace/data/BLSpeech/wavs/BL002-001-0001.wav',
            '/workspace/data/BLSpeech/wavs/BL002-001-0002.wav',
            '/workspace/data/BLSpeech/wavs/BL002-001-0003.wav'
]

his_audio = ['/workspace/data/BLSpeech/wavs/BL002-001-0001.wav',
            '/workspace/data/BLSpeech/wavs/BL002-001-0002.wav',
]
his_audio = ['/workspace/data/wenetspeech/wavs/X0000001792_27734668_S00016.wav',
             '/workspace/data/wenetspeech/wavs/X0000001792_27734668_S00017.wav']

tts.tts_to_file(text="hello, i am xiaoming.",
                file_path="/workspace/TTS/output/xtts-v2_1-01-17-1.wav",
                speaker_wav=his_audio,
                speaker_id=0,
                language="en")

# tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
#                 file_path="output/xtts-v2_2.wav",
#                 speaker_wav=["output/reference/p226_002_mic1.flac"],
#                 language="en")

print('')