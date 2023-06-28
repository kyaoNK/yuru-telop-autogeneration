import ffmpeg
import whisper

audio_filepath = '/home/nlp-lab/kojima/yuru_telop_autogeneration/audio/2023_01_22 (8)_食レポ5_編集なし_ショート.wav'

def transcribe(audio_filepath:str, model: whisper.model.Whisper, output_jsonpath:str) -> dict:
    result = model.transcribe(audio_filepath, verbose=True, language='ja')

model = whisper.load_model('large')
transcribe(audio_filepath=audio_filepath, model=model, output_jsonpath=output_jsonpath)
