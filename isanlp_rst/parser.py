from .dmrst_parser.predictor import Predictor as PredictorDMRST

class Parser:
    def __init__(self,
                 model_dir: str = None,
                 hf_model_name: str = 'tchewik/isanlp_rst_v3',
                 hf_model_version: str = None,
                 cuda_device: int = -1):

        if hf_model_version in ('gumrrg', 'rstdt', 'rstreebank'):
            self.predictor = PredictorDMRST(model_dir=model_dir,
                                            hf_model_name=hf_model_name,
                                            hf_model_version=hf_model_version,
                                            cuda_device=cuda_device)

        else:
            raise NotImplementedError

    def __call__(self, text):
        return self.predictor.parse_rst(text)
