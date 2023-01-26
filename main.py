from modelscope.pipelines import pipeline
from pathlib import Path
import os
import json

def get_files():
    """
    获取数据集中的所有文件信息
    :return: List[{path,name}]
    """
    audios = Path('/dataset').glob('./**/*.*')
    sources = []
    for audio_path in audios:
        audio_path = str(audio_path)
        sources.append({"path": audio_path, "name": os.path.basename(audio_path)})
    return sources


def main():
    """
    主函数
    :return:
    """
    p = pipeline('auto-speech-recognition',
                 'damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
    sources = get_files()
    for path, name in sources:
        f = open(f'/result/{name}.json','w',encoding='utf-8')
        result = p(path, )
        f.write(result['text'])
        f.close()

if __name__ == "__main__":
    main()