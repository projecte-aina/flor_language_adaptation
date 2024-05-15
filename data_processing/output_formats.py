from document import *
from typing import IO


def onion(
        document: Document,
        output_file: IO,
        i: int = None,
        ) -> None:
    output_file.write(f'<doc id={i} filename="{document.filepath}">\n')
    for paragraph in document:
        output_file.write("<p>\n")
        for sentence in paragraph:
            output_file.write("<s>\n")
            spans = sentence.word_spans
            moved_spans = sentence.word_spans[1:] + [(len(sentence), None)]
            for [(b1, _), (b2, _)] in zip(spans, moved_spans):
                output_file.write(sentence[b1:b2] + '\n')
            output_file.write("</s>\n")
        output_file.write("</p>\n")
    output_file.write("</doc>\n")


def default(
        document: Document,
        output_file: IO,
        i: int = None,
        ) -> None:
    for paragraph in document:
        for sentence in paragraph:
            output_file.write(sentence + " ")
            output_file.write('\n')
        output_file.write("\n")
    output_file.write(f'<end-of-doc> filename="{document.filepath}"\n')




