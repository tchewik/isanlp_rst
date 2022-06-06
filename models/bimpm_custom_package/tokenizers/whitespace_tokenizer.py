from typing import List

from allennlp.data.tokenizers import Token, Tokenizer
from overrides import overrides


@Tokenizer.register("whitespace_tokenizer")
class WhitespaceTokenizer(Tokenizer):
    def __init__(self, max_length=None) -> None:
        super().__init__()
        self.max_length = max_length

    def _tokenize(self, text):
        if self.max_length:
            return [Token(token) for token in text.split()][:self.max_length]
        return [Token(token) for token in text.split()]

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        tokens = self._tokenize(text)

        return tokens
