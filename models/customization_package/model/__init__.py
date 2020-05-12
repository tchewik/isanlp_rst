
try:
    from customization_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
    from customization_package.model.custom_bimpm import BiMpm
    from customization_package.model.custom_bimpm_predictor import CustomBiMPMPredictor
    from customization_package.model.contextual_bimpm import ContextualBiMpm
    from customization_package.model.contextual_bimpm_predictor import ContextualBiMpmPredictor
except ModuleNotFoundError:
    from models.customization_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
    from models.customization_package.model.custom_bimpm import BiMpm
    from models.customization_package.model.custom_bimpm_predictor import CustomBiMPMPredictor
    from models.customization_package.model.contextual_bimpm import ContextualBiMpm
    from models.customization_package.model.contextual_bimpm_predictor import ContextualBiMpmPredictor
