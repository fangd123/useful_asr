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

inference_pipeline = pipeline('auto-speech-recognition', 'damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch')

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


@app.post("/transcribe/")
async def transcribe_audio_file(file: UploadFile):
    # Save the uploaded audio file
    audio_bytes = await file.read()
    audio_path = f"{uuid.uuid4()}.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    # Load the audio file and extract features
    # 将语音文件载入内存
    sr = 16000
    waveform, samplerate = soundfile.read(audio_path)

    # Call the ASR model to transcribe the audio
    result = inference_pipeline(waveform, )
    # Clean up the temporary audio file
    os.remove(audio_path)
    print(result)
    if len(result['sentences']) == 1:
        return [{"text": result["text"], 'start': result['sentences'][0]["start"], 'end': result['sentences'][0]["end"]}]
    return result['sentences']


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)
