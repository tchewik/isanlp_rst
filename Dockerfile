FROM inemo/isanlp_base_cuda

RUN apt-get update
RUN apt-get install libffi-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y liblzma-dev

ENV PYENV_ROOT /opt/.pyenv
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
ENV PATH /opt/.pyenv/shims:/opt/.pyenv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
RUN pyenv install 3.7.4
RUN pyenv global 3.7.4

RUN pip install -U pip
RUN python -m pip install -U cython

RUN pip install dostoevsky
RUN dostoevsky download fasttext-social-network-model

RUN pip install setuptools==41.0.1 scipy scikit-learn==1.0.2 gensim==3.6.0 smart-open==1.7.0 tensorflow==2.1.0 keras h5py tensorflow-hub pandas nltk imbalanced-learn catboost==0.25.1

RUN pip install allennlp==2.9.3 allennlp-models==2.9.3
RUN pip install overrides

RUN pip install -U git+https://github.com/IINemo/isanlp.git

RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('omw-1.4')"

COPY src/isanlp_rst/td_rst_parser/src /src
COPY src/isanlp_rst /src/isanlp_rst
COPY pipeline_object.py /src/isanlp_rst/pipeline_object.py

COPY models/bimpm_custom_package models/bimpm_custom_package
COPY models/bimpm_custom_package bimpm_custom_package
COPY models/tf_idf models/tf_idf
COPY models/w2v models/w2v
COPY models/segmenter models/segmenter
COPY models/structure_predictor_baseline models/structure_predictor_baseline
COPY models/structure_predictor_bimpm models/structure_predictor_bimpm
COPY models/label_predictor_baseline models/label_predictor_baseline
COPY models/label_predictor_bimpm models/label_predictor_bimpm
COPY models/topdown_model models/topdown_model

### Uncomment this section if embedders are not in the current directory
## ELMo
#RUN curl -O http://vectors.nlpl.eu/repository/20/195.zip && unzip 195.zip && rm 195.zip
#RUN mkdir rsv_elmo && mv model.hdf5 rsv_elmo/model.hdf5 && mv options.json rsv_elmo/options.json
## fastText embeddings
#RUN curl -O http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_nltk_word_tokenize/ft_native_300_ru_wiki_lenta_nltk_word_tokenize.vec

COPY rsv_elmo /rsv_elmo

ENV PYTHONPATH=/src/isanlp_rst/
CMD [ "python", "/start.py", "-m", "pipeline_object", "-a", "create_pipeline", "--no_multiprocessing", "True"]
