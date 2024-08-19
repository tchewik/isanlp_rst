from .dmrst_parser.predictor import Predictor as PredictorDMRST

class Parser:
    AVAILABLE_VERSIONS = ('gumrrg', 'rstdt', 'rstreebank')

    def __init__(self,
                 model_dir: str = None,
                 hf_model_name: str = 'tchewik/isanlp_rst_v3',
                 hf_model_version: str = None,
                 cuda_device: int = -1):

        if hf_model_version in self.AVAILABLE_VERSIONS:
            self.predictor = PredictorDMRST(
                model_dir=model_dir,
                hf_model_name=hf_model_name,
                hf_model_version=hf_model_version,
                cuda_device=cuda_device
            )
        else:
            raise NotImplementedError(
                f"Available options for hf_model_version are: {', '.join(self.AVAILABLE_VERSIONS)}"
            )

    def __call__(self, text: str):
        return self.predictor.parse_rst(text)
