import json

import soundfile
import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from scipy.spatial.distance import cosine
import re
import soundfile

inference_sv_pipline = pipeline(
    task=Tasks.speaker_verification,
    model='damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch'
)


def short_sentence_split(text, text_postprocessed, timestamps):
    """
    输出每个短句的起始和结束时间戳
    :param text: 有标点符号的文本
    :param text_postprocessed: 未添加标点符号的词
    :param timestamps: 每个词的时间戳范围
    :return:
    """
    sentences = [x for x in re.split('[。？！，]', text) if x != '']  # 使用正则表达式将文本分割成短句
    have_sentences_words = []
    sentence_comma_index = []
    for sentence in sentences:
        # 在中文字符前后添加空格
        text = re.sub(r"([\u4e00-\u9fa5])", r" \1 ", sentence)
        sentence_words_list = [x for x in text.split() if x != '']
        have_sentences_words.extend(sentence_words_list)
        # 记录最后一个字符的所在位置，作为标点符号的位置
        sentence_comma_index.append(len(have_sentences_words) - 1)

    parttimes = []  # 存储每个短句的时间戳

    start_time, end_time = None, None
    i = 0
    for timestamp, word in zip(timestamps, have_sentences_words):
        if start_time is None:
            start_time = timestamp[0] / 1000
        end_time = timestamp[1] / 1000

        if i in sentence_comma_index:
            parttimes.append([start_time, end_time])
            start_time, end_time = None, None
        i += 1
    if start_time is not None:
        parttimes.append([start_time, end_time])
    return parttimes, sentences


class SpeakerIDGenerator:
    def __init__(self):
        self.next_id = 1

    def generate_new_speaker_id(self):
        """生成一个新的说话人ID"""
        self.next_id += 1
        return self.next_id


def split_audio_by_text_and_timestamp(audio_file, text_list, timestamp_list):
    """
    定义一个函数，用于将文本短语和时间戳转换为语音片段
    :param audio_file: 语音文件
    :param text_list: 文本列表
    :param timestamp_list: 时间戳列表
    :return: 语音片段列表
    """
    # 将语音文件载入内存
    sr = 16000
    waveform, samplerate = soundfile.read('audio.wav', dtype="int16")

    # 将时间戳和文本短语一一对应，生成语音片段列表
    audio_segments = []
    for i in range(len(text_list)):
        start_time = timestamp_list[i][0]
        end_time = timestamp_list[i][1]
        text = text_list[i]
        # 计算每个时间戳对应的采样点位置
        start_index = int((start_time + 0.1) * sr)
        end_index = int((end_time - 0.1) * sr)
        # 提取语音片段并进行预处理
        segment = waveform[start_index:end_index]
        soundfile.write(f'audio_part/{i}.wav', segment, samplerate)
        # 将语音片段信息加入到列表中
        audio_segments.append({
            "text": text,
            "start_time": start_time,
            "end_time": end_time,
            "audio": segment
        })
    return audio_segments


def speaker_diari(audio_segments):
    """
    说话人日志
    :param audio_segments: 语音片段列表词典
    :return: 包含说话人信息的语音片段列表词典
    """
    # 记录说话人的日志
    speaker_log = []

    # 处理第一个语音片段，将其视为新的说话人，并将其添加到日志中
    segment = audio_segments[0]
    text = segment["text"]
    start_time = segment["start_time"]
    end_time = segment["end_time"]
    audio = segment["audio"]
    embedding = inference_sv_pipline(audio_in=audio)["spk_embedding"]
    speaker_log.append({"text": text, "embedding": embedding, "id": 1})
    speaker_id_gen = SpeakerIDGenerator()

    # 遍历每个语音片段，确定其所属的说话人，并将其添加到日志中
    for segment in audio_segments:
        text = segment["text"]
        start_time = segment["start_time"]
        end_time = segment["end_time"]
        audio = segment["audio"]
        if audio.shape[0] == 0:
            continue

        embedding = inference_sv_pipline(audio_in=audio)["spk_embedding"]  # 利用说话人确认算法计算语音片段的 embedding
        speaker_id = None
        for speaker in speaker_log:
            sv_threshold = 0.9465
            same_cos = np.sum(embedding * speaker["embedding"]) / (
                        np.linalg.norm(embedding) * np.linalg.norm(speaker["embedding"]))
            same_cos = max(same_cos - sv_threshold, 0.0) / (1.0 - sv_threshold) * 100.0
            if same_cos > 60:  # 如果与已知说话人的相似度高
                # 将该语音片段归为该说话人
                speaker_id = speaker["id"]
                break

        # 如果找不到已知的相似说话人，则将该语音片段归为新说话人
        if speaker_id is None:
            speaker_id = speaker_id_gen.generate_new_speaker_id()
            speaker_log.append({"id": speaker_id, "embedding": embedding, "speaker_id": speaker_id})
        # 将该语音片段的信息添加到日志中
        segment['speaker_id'] = speaker_id

    return speaker_log


if __name__ == "__main__":
    with open("result.json", 'r', encoding='utf8') as f:
        result = json.load(f)

    parttimes, sentences = short_sentence_split(result['text'], result['text_postprocessed'], result['time_stamp'])
    audio_segments = split_audio_by_text_and_timestamp('audio.wav', sentences, parttimes)
    speaker_diari(audio_segments)
