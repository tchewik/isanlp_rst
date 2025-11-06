from typing import Sequence

from .dmrst_parser.predictor import PredictorDMRST
from .universal_parser.predictor import PredictorUniRST


class Parser:
    DMRST_PARSERS = ('gumrrg', 'rstdt', 'rstreebank')
    UNIVERSAL_PARSERS = ('rrtrrg', 'unirst')
    AVAILABLE_VERSIONS = DMRST_PARSERS + UNIVERSAL_PARSERS

    def __init__(self,
                 model_dir: str = None,
                 hf_model_name: str = 'tchewik/isanlp_rst_v3',
                 hf_model_version: str = None,
                 relinventory: str = None,   # for universal parsers
                 relinventory_idx: int = 0,  # for universal parsers
                 cuda_device: int = -1):

        if hf_model_version in self.DMRST_PARSERS:
            self.predictor = PredictorDMRST(
                model_dir=model_dir,
                hf_model_name=hf_model_name,
                hf_model_version=hf_model_version,
                cuda_device=cuda_device
            )
        elif hf_model_version in self.UNIVERSAL_PARSERS:
            self.predictor = PredictorUniRST(
                model_dir=model_dir,
                hf_model_name=hf_model_name,
                hf_model_version=hf_model_version,
                relinventory=relinventory,
                relinventory_idx=relinventory_idx,
                cuda_device=cuda_device
            )
        else:
            raise NotImplementedError(
                f"Available options for hf_model_version are: {', '.join(self.AVAILABLE_VERSIONS)}"
            )

    def __call__(self, text: str):
        return self.predictor.parse_rst(text)

    def from_edus(self, edus: Sequence[str]):
        """Parse a document using predefined EDUs."""

        return self.predictor.parse_from_edus(edus)
