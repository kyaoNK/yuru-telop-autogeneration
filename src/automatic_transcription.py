from inaSpeechSegmenter import Segmenter
import tensorflow
import ffmpeg
import whisper
import tensorflow as tf

AUDIO_PATH = '/home/nlp-lab/kojima/yuru_telop_autogeneration/audio/'
audio_filepath = AUDIO_PATH + '2023_01_22 (8)_食レポ5_編集なし.wav'
output_jsonpath = AUDIO_PATH + 'transcribe_{}.json'.format(audio_filepath.split('.')[0])


# whisper
# def transcribe(audio_filepath:str, model: whisper.model.Whisper, output_jsonpath:str) -> dict:
#     result = model.transcribe(audio_filepath, verbose=True, language='ja')

# model = whisper.load_model('large')
# transcribe(audio_filepath=audio_filepath, model=model, output_jsonpath=output_jsonpath)

# 有声区間検出
# seg = Segmenter(vad_engine='smn', detect_gender=False)
# segmentation = seg(audio_filepath)

# print(len(segmentation))
# for s in segmentation:
#     print(s)