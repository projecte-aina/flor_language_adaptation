from document import *
import input_formats 
import output_formats
from parser import Parser
from dataclasses import dataclass, field
from tqdm import tqdm


@dataclass
class CleaningArguments:
    input_path: str = field(
        metadata={"help": "Input data path"},
    )
    output_path: str = field(
        default="output",
        metadata={"help": "Output path."},
    )
    input_format: str = field(
        default="default",
        metadata={"help": "Read input in default format, see input_formats.py for more information."},
    )
    output_format: str = field(
        default="onion",
        metadata={"help": "Output in onion vertical format"},
    )
    no_filter: bool = field(
        default=False,
        metadata={"help": "Do not apply any filter"},
    )
    min_sentences_per_document: int = field(
        default=10,
        metadata={"help": "Minimum number of sentences per document."},
    )
    max_number_ellipsis_per_document: int = field(
        default=1,
        metadata={"help": "Filter documents containing more ellipsis."},
    )
    language: str = field(
        default=None,
        metadata={"help": "Filter documents are written in a different language (The language of a document is defined as the most common language of the sentences of all paragraphs)."},
    )
    min_number_of_words_per_paragraph: int = field(
        default=10,
        metadata={"help": "Filter paragraphs with less words."},
    )
    allowed_end_of_sentence: Optional[List[str]] = field(
        default_factory=lambda: ['!', '.', ':', ';', '?'],
        metadata={"help": "Filter sentences that do not end with one of the allowed characters"},
    )


def document_filter(document: Document, cleaning_args: CleaningArguments) -> bool:
    """ Given a Document return a boolean value indicating whether the document pass all filters or not """
    if document.num_sentences < cleaning_args.min_sentences_per_document:
        return False
    if document.count("...") > cleaning_args.max_number_ellipsis_per_document:
        return False
    if document.get_language() != cleaning_args.language:
        return False
    return True


def paragraph_filter(paragraph: Paragraph, cleaning_args: CleaningArguments) -> bool:
    """ """
    if paragraph.num_words < cleaning_args.min_number_of_words_per_paragraph:
        return False
    return True


def sentence_filter(sentence: Sentence, cleaning_args: CleaningArguments) -> bool:
    """ """
    for char in sentence:
        if char.isupper(): 
            break
        if char.isnumeric(): 
            break
        if char.islower():
            return False
    if sentence[-1] not in cleaning_args.allowed_end_of_sentence:
        return False
    return True


def main():
    parser = Parser(CleaningArguments)
    cleaning_args = parser.parse_args_into_dataclasses()[0]
    
    parse_input_file = getattr(input_formats, cleaning_args.input_format)
    parse_output_format = getattr(output_formats, cleaning_args.output_format)

    with open(cleaning_args.output_path, 'w') as output_file:

        for document in tqdm(parse_input_file(cleaning_args.input_path)):
            """
            Filtering happens in two steps: 
                1) with the input document, paragraph and sentence
                2) The reduced paragraphs (paragraphs that only contain the sentences that have passed the filters) 
                need to pass the paragraph filters again. 
                The reduced document (that consists on the paragraph that pass the filters) 
                needs to pass the document filters again.
            """
            if cleaning_args.no_filter:
                parse_output_format(document, output_file)
                continue
            # If the input document do not pass the document filters continue to next document).
            if not document_filter(document, cleaning_args):
                continue
            # otherwise create new document
            current_document = Document([], filepath=document.filepath)
            for paragraph in document:
                # if the paragraph do not pass the paragraph filters continue to next paragraph
                if not paragraph_filter(paragraph, cleaning_args):
                    continue
                # otherwise create new paragraph
                current_paragraph = Paragraph([])
                for sentence in paragraph:
                    if not sentence_filter(sentence, cleaning_args):
                        continue
                    current_paragraph.append(sentence)
                # the new paragraph has to pass the paragraph filters again
                if paragraph_filter(current_paragraph, cleaning_args):
                    current_document.append(current_paragraph)
            # the new document has to pass the document filters again.
            if document_filter(current_document, cleaning_args):
                # if the sentence, paragraph and document pass all filters then parse into the output format
                parse_output_format(document, output_file)


if __name__ == "__main__":
    main()
