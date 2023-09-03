import io
import os
import subprocess
import uuid
from typing import List
from fastapi.responses import FileResponse
import soundfile
from fastapi import FastAPI, File, UploadFile, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from pydub import AudioSegment

inference_pipeline = pipeline('auto-speech-recognition',
                              'damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
                              device='cpu')

sd_pipeline = pipeline(
    task='speaker-diarization',
    model='damo/speech_campplus_speaker-diarization_common',
    model_revision='v1.0.0'
)

app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:5432",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Sentence(BaseModel):
    text: str
    start: int
    end: int


class Transcript(BaseModel):
    sentences: List[Sentence]


def convert_to_wav(input_file, output_file):
    # 从任意格式加载音频文件
    audio = AudioSegment.from_file(input_file)

    # 设置采样率、声道和采样深度
    audio = audio.set_frame_rate(16000)  # 设置采样率为 16k
    audio = audio.set_channels(1)  # 设置为单声道

    # 导出为 WAV 格式
    audio.export(output_file, format="wav")


def merge_single_word_segments(output_data):
    merged_data = []

    i = 0
    while i < len(output_data):
        segment = output_data[i]

        # Check if it's a single-word segment and the next segment has the same speaker
        if len(segment["text"].strip()) == 1:
            merged_segment = {
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"],
                "speaker": segment["speaker"]
            }

            # Try to merge with subsequent segments
            while i + 1 < len(output_data) and len(output_data[i + 1]["text"].split()) == 1 and output_data[i + 1][
                "speaker"] == merged_segment["speaker"]:
                merged_segment["text"] += " " + output_data[i + 1]["text"]
                merged_segment["end"] = output_data[i + 1]["end"]
                i += 1

            merged_data.append(merged_segment)
        else:
            merged_data.append(segment)

        i += 1

    return merged_data


def find_speaker_for_ts(ts, speaker_data):
    for segment in speaker_data["text"]:
        if segment[0] <= ts <= segment[1]:
            return segment[2]
    return None


@app.post("/transcribe/")
async def transcribe_audio_file(file: UploadFile, distinct_speaker: bool = False, subtitle_file: bool = False):
    """

    :param file:
    :param distinct_speaker:
    :param subtitle_file:
    :return:
    """
    # Save the uploaded audio file
    audio_bytes = await file.read()
    temp_path = f"temp_{uuid.uuid4()}"
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    # Convert the audio to the desired format
    filename_stem = uuid.uuid4()
    audio_path = f"{filename_stem}.wav"

    convert_to_wav(temp_path, audio_path)
    os.remove(temp_path)  # remove the temporary file

    # Load the converted audio file and extract features
    sr = 16000
    waveform, samplerate = soundfile.read(audio_path)
    # Clean up the temporary audio file
    os.remove(audio_path)
    # Call the ASR model to transcribe the audio
    result = inference_pipeline(waveform)
    print(result)
    asr_data = result['sentences']
    if not distinct_speaker:
        if len(result['sentences']) == 1:
            return [
                {"text": result["text"], 'start': result['sentences'][0]["start"],
                 'end': result['sentences'][0]["end"]}]
        convert_to_srt(asr_data, f"{filename_stem}.srt")
        if subtitle_file:
            return FileResponse(f"{filename_stem}.srt",filename=f"{file.filename}.srt")
        return asr_data



    sd_result = sd_pipeline(waveform)

    for speaker_segment in sd_result["text"]:
        speaker_segment[0] = speaker_segment[0] * 1000
        speaker_segment[1] = speaker_segment[1] * 1000
        speaker_segment[2] = int(speaker_segment[2])

    output_data = []

    for segment in asr_data:
        previous_speaker = None
        text_list = segment['text_seg'].split()
        modified = False  # Track if the segment is split

        for idx, ts_segment in enumerate(segment['ts_list']):
            current_speaker = find_speaker_for_ts(ts_segment[0], sd_result)

            if previous_speaker is None:
                previous_speaker = current_speaker

            # If speaker changes in the middle of a segment
            if current_speaker != previous_speaker:
                modified = True
                new_seg = {
                    "text": ' '.join(text_list[:idx]),
                    "start": segment['start'],
                    "end": ts_segment[0],
                    "speaker": previous_speaker
                }
                output_data.append(new_seg)

                segment['start'] = ts_segment[0]
                text_list = text_list[idx:]

                previous_speaker = current_speaker

        # Append the last or only segment
        if modified:
            new_seg = {
                "text": ' '.join(text_list),
                "start": segment['start'],
                "end": segment['end'],
                "speaker": previous_speaker
            }
        else:
            # If not modified, use the original text value
            new_seg = {
                "text": segment['text'],
                "start": segment['start'],
                "end": segment['end'],
                "speaker": previous_speaker
            }
        output_data.append(new_seg)

    output_data = merge_single_word_segments(output_data)

    # Group by speaker while preserving the time order
    final_output = []

    current_speaker_data = None
    for item in output_data:
        if current_speaker_data and current_speaker_data["speaker"] == item["speaker"]:
            current_speaker_data["sentences"].append({
                "text": item["text"],
                "start": item["start"],
                "end": item["end"],
                "speaker": item["speaker"]
            })
        else:
            if current_speaker_data:
                final_output.append(current_speaker_data)
            current_speaker_data = {"speaker": item["speaker"], "sentences": [{
                "text": item["text"],
                "start": item["start"],
                "end": item["end"],
                "speaker": item["speaker"]
            }]}

    if current_speaker_data:
        final_output.append(current_speaker_data)

    print(final_output)

    return final_output


def extract_time(time_ms):
    """Extract time from milliseconds"""
    hour = time_ms // 3600000
    minute = (time_ms % 3600000) // 60000
    second = (time_ms % 60000) // 1000
    millisecond = time_ms % 1000

    return '{:02d}:{:02d}:{:02d},{:03d}'.format(hour, minute, second, millisecond)


def convert_to_srt(transcript, output):
    """Convert transcript to SRT format"""
    srt = ''
    # Generate SRT format
    for i, seg in enumerate(transcript, 1):
        start_time = extract_time(seg['start'])
        end_time = extract_time(seg['end'])
        text = seg['text']

        srt += f'{i}\n{start_time} --> {end_time}\n{text}\n\n'

    # Write to file
    with open(output, 'w', encoding='utf-8') as f:
        f.write(srt)

    print('SRT file generated.')


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)
