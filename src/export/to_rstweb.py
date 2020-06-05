import tempfile
from export.to_rs3 import ForestExporter
import requests


class RstWebExporter:
    """ Takes a document as a list of isanlp.annotation_rst.DiscourseUnit object
        and sends it as .rs3 file to the running rstweb service
        https://hub.docker.com/r/nlpbox/rstweb-service """
    
    def __init__(self, host='0.0.0.0', port=9000, project='rurstparser'):
        self._host = host
        self._port = port
        self._project = project
        self._doc_to_rs3 = ForestExporter(encoding='utf8')
        
    def __call__(self, document):
        with tempfile.NamedTemporaryFile(suffix='.rs3') as t:
            self._doc_to_rs3(document, t.name)
            files = {'rs3_file': (t.name, open(t.name, 'r'))}
            requests.post(f'http://{self._host}:{self._port}/api/documents/{self._project}/{t.name}', files=files)
