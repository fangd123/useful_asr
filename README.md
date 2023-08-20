# useful_asr

---

[English Version](README_EN.md)

[知乎文章介绍](https://zhuanlan.zhihu.com/p/651156659)

### 语音转录服务

这是一个基于 FastAPI 的语音转录服务。用户可以上传音频文件，服务会将其转换为文本。

#### 主要功能

1. 基于 FastAPI 框架。
2. 支持跨域请求。
3. 使用 `modelscope.pipelines` 进行语音识别。
4. 提供 `/transcribe/` 端点以上传音频文件并获取转录文本。

#### 如何运行

**注：首次启动程序会从modelsope仓库拉取模型文件存储至本地，需要几分钟的时间。**

##### 使用 Python

1. 确保您已安装所有库：`pip install -r requirements.txt`。
2. 运行 `app.py`：`python app.py`。
3. 服务将在 `0.0.0.0` 上的端口 `8100` 上启动。使用你的浏览器或 API 工具访问。

##### 使用 Docker

###### 直接拉取

```bash
docker pull registry.cn-guangzhou.aliyuncs.com/fangd123/asr:1.0
# 若无GPU
docker run --name asr -p 8100:8100 registry.cn-guangzhou.aliyuncs.com/fangd123/asr:1.0
# 若有GPU（需要先安装NVIDIA Container Toolkit）
docker run --name asr --runtime=nvidia --gpus all -p 8100:8100 registry.cn-guangzhou.aliyuncs.com/fangd123/asr:1.0
```

###### 源码构建
1. 构建 Docker 镜像：`docker build -t asr-service .`
2. 运行 Docker 容器：`docker run -p 8100:8100 asr-service`
3. 与上面的 Python 方法一样，服务将在 `0.0.0.0` 上的端口 `8100` 上启动。

#### 使用方法

1. 使用 POST 请求访问 `/transcribe/` 端点并上传您的音频文件。
2. 服务将返回转录的文本。

注：可访问`/docs`端点进入调试界面

#### API 返回格式

服务返回一个包含以下字段的 JSON 数组：

- `text`：转录的文本。
- `start`：文本的开始时间（单位：毫秒）。
- `end`：文本的结束时间（单位：毫秒）。
- `text_seg`：分词文本。
- `ts_list`：每段文本的时间戳列表。

例如：

```json
[
  {
    "text": " just time 哎，",
    "start": 4090,
    "end": 6470,
    "text_seg": "just time 哎 ",
    "ts_list": [
      [4090, 4330],
      [4470, 4875],
      [6250, 6470]
    ]
  },
  ...
]
```