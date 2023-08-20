import io
import os
import subprocess
import uuid
from typing import List

import soundfile
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from pydub import AudioSegment

inference_pipeline = pipeline('auto-speech-recognition',
                              'damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch')

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


@app.post("/transcribe/")
async def transcribe_audio_file(file: UploadFile):
    # Save the uploaded audio file
    audio_bytes = await file.read()
    temp_path = f"temp_{uuid.uuid4()}"
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    # Convert the audio to the desired format
    audio_path = f"{uuid.uuid4()}.wav"
    convert_to_wav(temp_path, audio_path)
    os.remove(temp_path)  # remove the temporary file

    # Load the converted audio file and extract features
    sr = 16000
    waveform, samplerate = soundfile.read(audio_path)

    # Call the ASR model to transcribe the audio
    result = inference_pipeline(waveform)

    # Clean up the temporary audio file
    os.remove(audio_path)
    print(result)
    if len(result['sentences']) == 1:
        return [
            {"text": result["text"], 'start': result['sentences'][0]["start"], 'end': result['sentences'][0]["end"]}]
    return result['sentences']


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)
