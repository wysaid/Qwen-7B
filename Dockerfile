# python 3.8 and above
# pytorch 1.12 and above, 2.0 and above are recommended
# CUDA 11.4 and above are recommended (this is for GPU users, flash-attention users, etc.)

# based on modelscope docker image
# registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.8.0
# registry.cn-beijing.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.8.0
FROM registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.8.0

ARG workdir=/var/app
RUN mkdir -p ${workdir}

RUN git lfs install

WORKDIR ${workdir}
COPY requirements.txt requirements_openai_api.txt requirements_web_demo.txt ./

# Install Qwen dependencies
RUN pip install -r requirements.txt
WORKDIR ${workdir}
RUN git clone -b v1.0.8 https://github.com/Dao-AILab/flash-attention && \
    cd flash-attention && \
    pip install . && \
    pip install csrc/layer_norm && \
    pip install csrc/rotary

# Install OpenAI API dependencies
WORKDIR ${workdir}
RUN pip install -r requirements_openai_api.txt

# Install webUI dependencies
WORKDIR ${workdir}
RUN pip install -r requirements_web_demo.txt

# Install AutoGPTQ (for Qwen/Qwen-7B-Chat-Int4)
ENV BUILD_CUDA_EXT=0
WORKDIR ${workdir}
RUN git clone https://github.com/PanQiWei/AutoGPTQ.git && \
    cd AutoGPTQ && \
    pip install .
# RUN pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu117/

# clone Qwen-7B-Chat-Int4 model
RUN git clone https://huggingface.co/Qwen/Qwen-7B-Chat-Int4
# clone Qwen-7B-Chat model
# RUN git clone https://huggingface.co/Qwen/Qwen-7B-Chat

# set TZ, make logs dir, and expose port 8080
ENV TZ=Asia/Shanghai
RUN mkdir -p ${workdir}/logs && chmod 777 ${workdir}/logs
VOLUME /var/app/logs
RUN chmod -R 755 ${workdir}
EXPOSE 8080

# copy main app
WORKDIR ${workdir}
# COPY cache_model.py openai_api_web.py ./
COPY openai_api_web.py ./

# set cache dir
# ENV TRANSFORMERS_CACHE=${workdir}/cache/huggingface/transformers/
# ENV HF_HOME=${workdir}/cache/huggingface/
# ENV MPLCONFIGDIR=${workdir}/.config/matplotlib/

# Cache model
# RUN python3 cache_model.py -c Qwen/Qwen-7B-Chat-Int4
# RUN python3 cache_model.py

# RUN adduser -g -u -D 20001 appuser
RUN useradd -r -m appuser -u 20001 -g 0
RUN chown -R 20001 ${workdir}

# Offline mode, check https://huggingface.co/docs/transformers/v4.15.0/installation#offline-mode
ENV HF_DATASETS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

CMD ["python3", "openai_api_web.py", "-c", "./Qwen/Qwen-7B-Chat-Int4"]
# CMD ["python3", "openai_api_web.py", "-c", "Qwen/Qwen-7B-Chat-Int4"]
# CMD ["python3", "openai_api_web.py"]
# ENTRYPOINT [ "/bin/bash" ]
