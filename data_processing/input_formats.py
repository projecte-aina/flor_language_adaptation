from document import *
from tqdm import tqdm
from typing import Iterator
from nltk.tokenize import sent_tokenize


def default(path: str) -> Iterator[Document]:
    """
    Parse documents with blank lines between paragraphs and documents separated by <end-of-doc>
    """

    with open(path) as f:
        current_document = Document([])
        current_paragraph = Paragraph([])
        for line in tqdm(f):
            line = line[:-1]
            if line[:12] == "<end-of-doc>":
                current_document.append(current_paragraph)
                yield current_document
                current_paragraph = Paragraph([])
                current_document = Document([], filepath=path)
            elif line == "":
                current_document.append(current_paragraph)
                current_paragraph = Paragraph([])
            else:
                for sentence in sent_tokenize(line):
                    try:
                        current_sentence = Sentence(sentence)
                        current_paragraph.append(current_sentence)
                    except:
                        pass


def onion(path: str) -> Iterator[Document]:
    """
    Parse documents like:

    """
    with open(path) as f:
        current_document = None
        current_paragraph = None
        current_sentence = ""
        for line in tqdm(f):
            line = line[:-1]
            if line[:4] == "<doc":
                _, id_, filepath = line.split(' ')
                filepath = filepath.split('\"')[1]
                current_document = Document([], filepath=filepath)
            elif line == '<p>':
                current_paragraph = Paragraph([])
            elif line == '</p>':
                current_document.append(current_paragraph)
                current_paragraph = None
            elif line == '</doc>':
                yield current_document
                current_document = None
            elif line == "<s>":
                current_sentence = ""
            elif line == "</s>":
                try: 
                    current_paragraph.append(Sentence(current_sentence))
                except:
                    pass
                current_sentence = ""
            else:
                current_sentence += line  


def cawac(path: str) -> Iterator[Document]:
    """
    Cawarg format
    """
    with open(path) as f:
        current_document = None
        for line in tqdm(f):
            line = line[:-1]
            if line[:4] == "<doc":
                current_document = Document([], filepath=path)
            if line == "</doc>":
                yield current_document
                current_document = None
            if line[:2] == "<p" and line[-4:] == "</p>":
                current_paragraph = Paragraph([])
                for sentence in sent_tokenize(line[15:-4]):
                    try: 
                        current_sentence = Sentence(sentence)
                        current_paragraph.append(current_sentence)
                    except:
                        pass
                current_document.append(current_paragraph)





        

