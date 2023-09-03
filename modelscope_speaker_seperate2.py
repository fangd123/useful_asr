from modelscope.pipelines import pipeline
import json
sd_pipeline = pipeline(
    task='speaker-diarization',
    model='damo/speech_campplus_speaker-diarization_common',
    model_revision='v1.0.0'
)
input_wav = r'C:\Users\fang_\Downloads\史蒂夫说348期 - 心理科普：得抑郁症了，我该怎么办？身边人该怎么办？.wav'
result = sd_pipeline(input_wav)
print(result)

