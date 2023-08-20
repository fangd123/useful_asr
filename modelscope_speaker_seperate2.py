from modelscope.pipelines import pipeline
sd_pipeline = pipeline(
    task='speaker-diarization',
    model='damo/speech_campplus_speaker-diarization_common',
    model_revision='v1.0.0'
)
input_wav = '1.wav'
result = sd_pipeline(input_wav)
print(result)
# 如果有先验信息，输入实际的说话人数，会得到更准确的预测结果
result = sd_pipeline(input_wav, oracle_num=4)
print(result)