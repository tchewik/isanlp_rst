
try:
    from customization_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
    from customization_package.model.custom_bimpm import BiMpm
    from customization_package.model.custom_bimpm_predictor import CustomBiMPMPredictor
except ModuleNotFoundError:
    from models.customization_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
    from models.customization_package.model.custom_bimpm import BiMpm
    from models.customization_package.model.custom_bimpm_predictor import CustomBiMPMPredictor
