from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

text = '现在年轻人想法真的不一样了，我公司好几个告诉我不想结婚，男的打游戏女的追爱豆，电脑手机屏都是不认识的小鲜肉。有个女的告诉我，她以后赚到钱了，想去很远的地方，一个人生活，我说那你有想好怎么融入当地吗？她说我什么要融入，我很奇怪，那不是很孤独吗？她说我现在也是一个人在杭州，没有任何社交，不孤独啊。我想了想，也有道理，我们这代人依靠社交获得信息，找工作，做生意，谈恋爱都离不开社交。现在网络发达，年轻人不需要通过社交获得信息了，也是挺可怕的一件事。她还说，最讨厌回家了，逢年过节，没完没了的走亲戚，要么接待亲戚，太累了。我也不知道说什么好了。感觉互联网已经开始改变我们的社会结构。'

model_id = 'damo/speech_sambert-hifigan_tts_zh-cn_16k'
sambert_hifigan_tts = pipeline(task=Tasks.text_to_speech, model=model_id)
output = sambert_hifigan_tts(input=text, voice='zhibei_emo')
wav = output[OutputKeys.OUTPUT_WAV]
with open('output.wav', 'wb') as f:
    f.write(wav)
