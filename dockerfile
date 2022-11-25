FROM python:3.9.15-slim
ENV LANG en_US.UTF-8 LC_ALL=en_US.UTF-8
COPY . /mbvits
RUN cd /mbvits   \
    && pip install --no-cache-dir -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/ \
    && python download_nltk.py
ENV PYTHONIOENCODING=utf-8
WORKDIR /mbvits
ENTRYPOINT ["python","inference_api.py"]