from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

p = pipeline('auto-speech-recognition', 'damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch')

p('http://www.modelscope.cn/api/v1/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/repo?Revision=master\u0026FilePath=example/asr_example.wav',)