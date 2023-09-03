from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import json
p = pipeline('auto-speech-recognition', 'damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
result = p('1.wav',)
print(json.dumps(result['sentences'],ensure_ascii=False))
