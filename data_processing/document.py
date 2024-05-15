from __future__ import annotations
from typing import List
from typing import Optional
from nltk.tokenize import sent_tokenize, TreebankWordTokenizer
import nltk
import fasttext
from copy import copy

# nltk.data.path.append('/gpfs/projects/bsc88/tools/simple_cleaner/nltk_data')  # TODO: remove path
model = fasttext.load_model("lid.176.bin")


class NotSameFilepathError(Exception):
    """ Raised when trying to combine documents from different files. """
    def __init__(self, 
            message: str ="Trying to combine two documents with different filepaths.") -> Exception:
        super().__init__(message)


def most_common(l):
    if len(l) > 0:
        return max(set(l), key=l.count)
    else:
        return None


class Sentence(str):
    def __init__(self,
            text: str,
            language: Optional[str] = None) -> None:
        self.text = text 
        self.word_spans = list(TreebankWordTokenizer().span_tokenize(text))
        self.num_words = len(self.word_spans)
        if language is None:
            self.language = model.predict(text, k=1)[0][0][-2:]


class Paragraph(list):
    """ 
    A paragraph is defined as a list of Sentences
    """
    def __init__(self, 
            sentences: List[Sentence]
            ) -> None:

        super().__init__(sentences)
        self.num_words = sum([sentence.num_words for sentence in sentences])
        self.num_sentences = len(sentences)
        self.cached_language = most_common([s.language for s in self])

    def get_language(self) -> str:
        if self.cached_language is None:
            self.cached_language = most_common([s.language for s in self])
        return self.cached_language

    def append(self,
            sentence: Sentence) -> None:
        self.num_words += sentence.num_words
        self.num_sentences += 1
        self.cached_language = None
        super().append(sentence)

    def clear(self) -> None:
        self.num_words = 0
        self.num_sentences = 0
        self.cached_language = None
        super().clear()

    def copy(self) -> Paragraph:
        return Paragraph(copy(self))

    def count(self, pattern: str) -> int:
        """ count how many times a pattern appears in any of the sentences of the paragraph """
        return sum([sentence.count(pattern) for sentence in self])

    def extend(self,
            sentences: List[Sentence]) -> None:
        for sentence in sentences:
            self.append(sentence)

    def insert(self,
            index: int, 
            sentence: Sentence) -> None:
        num_old_sentence_words = self[index].num_words
        self.num_words -= num_old_sentence_words
        self.num_words += sentence.num_words
        self.cached_language = None
        self[index] = sentence

    def pop(self,
            index: int) -> Sentence:
        poped_sentence = super().pop(index)
        self.num_words -= poped_sentence.num_words
        self.num_sentences -= 1
        self.cached_language = None
        return poped_sentence

    def remove(self,
            sentence: Sentence) -> None:
        removed_sentence = self[super().index(sentence)]
        self.num_words -= removed_sentence.num_words
        self.num_sentences -= 1
        self.cached_language = None
        super().remove(sentence)

    def __add__(self,
            paragraph: Paragraph) -> Paragraph:
        return_p = Paragraph([])
        return_p.extend(self)
        return_p.extend(paragraph)
        return return_p

    def __iadd__(self,
            paragraph: Paragraph) -> Paragraph:
        self.extend(paragraph)
        return self


class Document(list):
    """ 
    A Document is defined as a list of Paragraphs 
    """
    def __init__(self,
            paragraphs: List[Paragraph],
            filepath: Optional[str] = None,
            ) -> None:
        super(Document, self).__init__(paragraphs)
        self.num_words = sum([paragraph.num_words for paragraph in paragraphs])
        self.num_sentences = sum([paragraph.num_sentences for paragraph in paragraphs])
        self.filepath = filepath
        self.cached_language = most_common([s.language for p in self for s in p])

    def get_language(self) -> str:
        if self.cached_language is None:
            self.cached_language = most_common([s.language for p in self for s in p])
        return self.cached_language

    def append(self,
            paragraph: Paragraph) -> None:
        self.num_words += paragraph.num_words
        self.num_sentences += paragraph.num_sentences
        self.cached_language = None
        super().append(paragraph)

    def clear(self) -> None:
        self.num_words = 0
        self.num_sentences = 0
        self.filepath = None
        self.cached_language = None
        super().clear()

    def copy(self) -> Document:
        return Document(copy(self))

    def count(self, pattern: str) -> int:
        """ count how many times a pattern appears in any of the paragraphs of the document """
        return sum([paragraph.count(pattern) for paragraph in self])

    def extend(self,
            paragraphs: List[Paragraph]) -> None:
        for paragraph in paragraphs:
            self.append(paragraph)

    def insert(self,
            index: int, 
            paragraph: Paragraph) -> None:
        num_old_sentence_words = self[index].num_words
        self.num_words -= num_old_sentence_words
        self.num_words += sentence.num_words
        self.cached_language = None
        self[index] = paragraph

    def pop(self,
            index: int) -> Paragraph:
        poped_paragraph = super().pop(index)
        self.num_words -= poped_paragraph.num_words
        self.num_sentences -= poped_paragraph.num_sentences
        self.cached_language = None
        return poped_paragraph

    def remove(self,
            paragraph: Paragraph) -> None:
        removed_paragraph = self[super().index(paragraph)]
        self.num_words -= removed_paragraph.num_words
        self.num_sentences -= removed_paragraph.num_sentences
        self.cached_language = None
        super().remove(paragraph)

    def __add__(self,
            document: Document) -> Document:
        if self.filepath != document.filepath:
            raise NotSameFilepathError()
        return_doc = Document([])
        return_doc.extend(self)
        return_doc.extend(document)
        return return_doc

    def __iadd__(self,
            document: Document) -> Document:
        if self.filepath != document.filepath:
            raise NotSameFilepathError()
        self.extend(document)
        return self




