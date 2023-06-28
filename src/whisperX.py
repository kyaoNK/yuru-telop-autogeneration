import whisperx
import ffmpeg
import os

device = 'cuda'
HOME_DIRPATH = '/home/nlp-lab/kojima/yuru_telop_autogeneration/'
def create_whisper(model_name='large-v2'):
    model = whisperx.load_model('large-v2', device)
    return model

def transcribe(model, audio_filepath):
    output_filepath = HOME_DIRPATH + 'txt/' + os.path.splitext(os.path.basename(audio_filepath))[0]
    print(f'{audio_filepath}\n->{output_filepath}')

    result = model.transcribe(audio_filepath)

    model_a , metadata = whisperx.load_align_model(language_code=result['language'], device=device)
    result_aligned = whisperx.align(result['segments'], model_a, metadata, audio_filepath, device)

    write(output_filepath, result, False, 'segments')  # whisper
    write(output_filepath, result_aligned, True, 'segments') # whipserX
    write(output_filepath, result_aligned, True, 'word_segments') # whisperX forced-alignment

    return result, result_aligned

def write(output_filepath, result, is_whisperX, seg):
    whisper_name = 'whipserX' if is_whisperX else 'whisper'
    with open(output_filepath + '_' + whisper_name + '_'  + seg + '.srt', 'w') as f:
        whisperx.utils.write_srt(result[seg], f)

    with open(output_filepath + '_' + whisper_name + '_' + seg + '.vtt', 'w') as f:
        whisperx.utils.write_vtt(result[seg], f)

    with open(output_filepath + '_' + whisper_name + '_' + seg + '.txt', 'w') as f:
        whisperx.utils.write_vtt(result[seg], f)

model_name = 'large-v2'
model = create_whisper(model_name)
sample_filepath = HOME_DIRPATH + 'audio/2023_01_22 (8)_食レポ5_編集なし_ショート.wav'
transcribe(model, sample_filepath)
