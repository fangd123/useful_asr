# 使用pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime的镜像作为基本镜像
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 设置时区为中国
#RUN apt-get update && apt-get install -y tzdata
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 设置工作文件夹为/app
WORKDIR /app

# 将当前目录下的所有文件拷贝到app目录中
COPY . /app

# 替换pip源为清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装requirements.txt并清除pip缓存
RUN pip install --no-cache-dir -r requirements.txt

# 运行app.py文件
CMD ["python", "app.py"]