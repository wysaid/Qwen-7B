# python 3.8 and above
# pytorch 1.12 and above, 2.0 and above are recommended
# CUDA 11.4 and above are recommended (this is for GPU users, flash-attention users, etc.)

# based on modelscope docker image
# registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.8.0
# registry.cn-beijing.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.8.0
FROM registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.8.0

RUN mkdir -p /data/shared/Qwen-7B
WORKDIR /data/shared/Qwen-7B/

# Users can also mount '/data/shared/Qwen-7B/' to keep the data
ADD . ./

# Install Qwen dependencies
RUN pip install transformers==4.31.0 accelerate tiktoken einops
RUN git clone -b v1.0.8 https://github.com/Dao-AILab/flash-attention && \
    cd flash-attention && \
    pip install . && \
    pip install csrc/layer_norm && \
    pip install csrc/rotary
RUN cd /data/shared/Qwen-7B

# Install AutoGPTQ (for Qwen/Qwen-7B-Chat-Int4)
RUN git clone https://github.com/PanQiWei/AutoGPTQ.git && \
    cd AutoGPTQ && \
    pip install .
RUN cd /data/shared/Qwen-7B

# Install webUI dependencies
RUN pip install -r requirements_web_demo.txt

# Install OpenAI API dependencies
RUN pip install fastapi uvicorn openai pydantic sse_starlette

