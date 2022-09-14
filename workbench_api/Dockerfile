FROM alpine:3.8

RUN apk add linux-headers
RUN apk update && apk add gcc g++ && apk add git python3 python3-dev py3-pip py3-setuptools
RUN pip3 install -U pip
RUN pip3 install markupsafe flask git+https://github.com/IINemo/isanlp.git

WORKDIR /opt
COPY isanlp_rst_api.py .
COPY parser_wrapper.py .
COPY output_file.txt .

CMD ["python3", "isanlp_rst_api.py"]