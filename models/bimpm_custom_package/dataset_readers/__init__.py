
try:
    from bimpm_custom_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
    from bimpm_custom_package.dataset_readers.custom_reader import CustomDataReader
except ModuleNotFoundError:
    from models.bimpm_custom_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
    from models.bimpm_custom_package.dataset_readers.custom_reader import CustomDataReader
