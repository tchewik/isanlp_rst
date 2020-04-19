
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers import Token, Tokenizer, CharacterTokenizer, WordTokenizer
from overrides import overrides
from typing import Dict, List


@Tokenizer.register("simple")
class WhitespaceTokenizer(Tokenizer):
    def __init__(self) -> None:
        super().__init__()

    def _tokenize(self, text):
        return [Token(token) for token in text.split()]

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        tokens = self._tokenize(text)

        return tokens
