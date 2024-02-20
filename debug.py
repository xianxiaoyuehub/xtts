import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2",
        "/workspace/pretrained/tts/tts_models--multilingual--multi-dataset--xtts_v2-0130",
        gpu='cuda')

tts.synthesizer.tts_config.temperature = 0.1
tts.synthesizer.tts_config.top_k = 20
tts.synthesizer.tts_config.top_p = 0.1

# his_audio = ['/workspace/data/BLSpeech/wavs/BL002-001-0001.wav',
#             '/workspace/data/BLSpeech/wavs/BL002-001-0002.wav',
#             '/workspace/data/BLSpeech/wavs/BL002-001-0003.wav'
# ]

# his_audio = ['/workspace/data/BLSpeech/wavs/BL002-001-0001.wav',
#             '/workspace/data/BLSpeech/wavs/BL002-001-0002.wav',
# ]

his_audio = ['/workspace/data/wenetspeech/wavs/X0000001792_27734668_S00016.wav',
             '/workspace/data/wenetspeech/wavs/X0000001792_27734668_S00017.wav']
# X0000001862_27414385_S00592
# import pdb
# pdb.set_trace()

tts.tts_to_file(text="hello, i am xiaoming.",
                file_path="/workspace/TTS/output/xtts-v2_1-01-19-1.wav",
                speaker_wav=his_audio,
                speaker_id=0,
                language="en")

print('')