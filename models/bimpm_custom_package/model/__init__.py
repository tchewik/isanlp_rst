
try:
    from bimpm_custom_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
    from bimpm_custom_package.model.custom_bimpm import BiMpm as CustomBiMpm
    from bimpm_custom_package.model.multiclass_bimpm import BiMpm as MulticlassBiMpm
    from bimpm_custom_package.model.custom_bimpm_predictor import CustomBiMPMPredictor
except ModuleNotFoundError:
    from models.bimpm_custom_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
    from models.bimpm_custom_package.model.custom_bimpm import BiMpm as CustomBiMpm
    from models.bimpm_custom_package.model.multiclass_bimpm import BiMpm as MulticlassBiMpm
    from models.bimpm_custom_package.model.custom_bimpm_predictor import CustomBiMPMPredictor
