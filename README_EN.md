# useful_asr

---

### Voice Transcription Service

This is a voice transcription service based on FastAPI. Users can upload audio files, and the service will transcribe them into text.

#### Key Features

1. Built with the FastAPI framework.
2. Supports Cross-Origin Resource Sharing (CORS).
3. Utilizes `modelscope.pipelines` for automatic speech recognition.
4. Offers `/transcribe/` endpoint to upload audio files and receive transcriptions.

#### How to Run

**Note: On the first launch, the program will pull the model file from the modelsope repository and store it locally, which might take a few minutes.**

##### Using Python

1. Ensure you have installed all required libraries: `pip install -r requirements.txt`.
2. Run `app.py`: `python app.py`.
3. The service will be started on `0.0.0.0` at port `8100`. Access it using your browser or API tools.

##### Using Docker

###### Direct Pull

```bash
docker pull registry.cn-guangzhou.aliyuncs.com/fangd123/asr:1.0
# If no GPU is available
docker run --name asr -p 8100:8100 registry.cn-guangzhou.aliyuncs.com/fangd123/asr:1.0
# If GPU is available (requires prior installation of NVIDIA Container Toolkit)
docker run --name asr --runtime=nvidia --gpus all -p 8100:8100 registry.cn-guangzhou.aliyuncs.com/fangd123/asr:1.0
```

###### Build from Source

1. Build the Docker image: `docker build -t asr-service .`
2. Run the Docker container: `docker run -p 8100:8100 asr-service`
3. Similar to the Python method above, the service will start on `0.0.0.0` at port `8100`.

#### How to Use

1. Make a POST request to the `/transcribe/` endpoint and upload your audio file.
2. The service will return the transcribed text.

Note: Visit the `/docs` endpoint for the debugging interface.

#### API Response Format

The service returns a JSON array containing the following fields:

- `text`: Transcribed text.
- `start`: Start time of the text (in milliseconds).
- `end`: End time of the text (in milliseconds).
- `text_seg`: Segmented text.
- `ts_list`: Timestamp list for each segment of text.

Example:

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