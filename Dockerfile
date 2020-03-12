FROM inemo/isanlp_base_cuda

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y liblzma-dev

ENV PYENV_ROOT /opt/.pyenv
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
ENV PATH /opt/.pyenv/shims:/opt/.pyenv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
RUN pyenv install 3.6.4
RUN pyenv global 3.6.4

RUN pip install -U pip
RUN python -m pip install -U cython

RUN pip install dostoevsky
RUN dostoevsky download fasttext-social-network-model

RUN pip install setuptools==41.0.1 scipy scikit-learn==0.22.1 gensim==3.6.0 smart-open==1.7.0 tensorflow==2.1.0 keras h5py tensorflow-hub pandas nltk imbalanced-learn catboost

RUN pip install allennlp==0.9.0
RUN pip install -U git+https://github.com/IINemo/isanlp.git@discourse

RUN python -c "import nltk; nltk.download('stopwords')"

COPY src/isanlp_rst /src/isanlp_rst
COPY pipeline_object.py /src/isanlp_rst/pipeline_object.py
COPY models /models

RUN curl -O http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_pt.tar.gz && tar -xzvf rubert_cased_L-12_H-768_A-12_pt.tar.gz
RUN python -c "from allennlp.predictors import Predictor; predictor = Predictor.from_path('models/tony_segmentator/model.tar.gz')"

ENV PYTHONPATH=/src/isanlp_rst/
CMD [ "python", "/start.py", "-m", "pipeline_object", "-a", "create_pipeline", "--no_multiprocessing", "True"]