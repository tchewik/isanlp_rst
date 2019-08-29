FROM inemo/isanlp_base:0.0.5

RUN pip install -U pip
RUN python -m pip install -U cython

RUN pip install setuptools==41.0.1 scipy scikit-learn==0.20.2 gensim==3.6.0 smart-open==1.7.0 tensorflow==1.12.0 keras h5py tensorflow-hub pandas nltk imbalanced-learn catboost

RUN python -c "import tensorflow as tf; print(tf.__version__)"
RUN python -c "import nltk; nltk.download('stopwords')"

COPY src/isanlp_rst /src/isanlp_rst
COPY pipeline_object.py /src/isanlp_rst/pipeline_object.py
COPY models /models

COPY src/isanlp_rst/annotation_rst_pb2.py /root/.pyenv/versions/3.6.0/lib/python3.6/site-packages/isanlp/annotation_rst_pb2.py
COPY src/isanlp_rst/annotation.py /root/.pyenv/versions/3.6.0/lib/python3.6/site-packages/isanlp/annotation_rst.py
COPY src/isanlp_rst/annotation_to_protobuf.py /root/.pyenv/versions/3.6.0/lib/python3.6/site-packages/isanlp/annotation_to_protobuf.py

ENV PYTHONPATH=/src/isanlp_rst/

CMD [ "python", "/start.py", "-m", "pipeline_object", "-a", "PPL_RST" ]
