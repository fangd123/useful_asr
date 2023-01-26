from io import StringIO
from typing import Iterator

from modelscope.pipelines import pipeline
from pathlib import Path
import os
import json
from utils import *


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


def write_result(result: dict, source_name: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    text = result["text"]
    languageMaxLineWidth = 40

    print("Max line width " + str(languageMaxLineWidth))
    vtt = get_subs(result["segments"], "vtt", languageMaxLineWidth)
    srt = get_subs(result["segments"], "srt", languageMaxLineWidth)

    output_files = []
    output_files.append(create_file(srt, output_dir, source_name + "-subs.srt"))
    output_files.append(create_file(vtt, output_dir, source_name + "-subs.vtt"))
    output_files.append(create_file(text, output_dir, source_name + "-transcript.txt"))

    return output_files, text, vtt


def create_file(text: str, directory: str, fileName: str) -> str:
    # Write the text to a file
    with open(os.path.join(directory, fileName), 'w+', encoding="utf-8") as file:
        file.write(text)

    return file.name


def get_subs(self, segments: Iterator[dict], format: str, maxLineWidth: int) -> str:
    segmentStream = StringIO()

    if format == 'vtt':
        write_vtt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
    elif format == 'srt':
        write_srt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
    else:
        raise Exception("Unknown format " + format)

    segmentStream.seek(0)
    return segmentStream.read()


def main():
    """
    主函数
    :return:
    """
    p = pipeline('auto-speech-recognition',
                 'damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
    sources = get_files()
    for one in sources:
        path = one['path']
        name = one['name']
        f = open(f'/result/{name}.json','w',encoding='utf-8')
        print(path)
        result = p(path, )
        text = result['text']
        new_result = {}
        new_result['text_punc'] = text['text_punc']
        new_result['time_stamp'] = text['time_stamp']
        f.write(json.dumps(new_result,ensure_ascii=False))
        f.write('\n')


if __name__ == "__main__":
    main()
