
try:
    from customization_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
    from customization_package.dataset_readers.custom_reader import CustomDataReader
    from customization_package.dataset_readers.contextual_reader import ContextualReader
except ModuleNotFoundError:
    from models.customization_package.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
    from models.customization_package.dataset_readers.custom_reader import CustomDataReader
    from models.customization_package.dataset_readers.contextual_reader import ContextualReader
