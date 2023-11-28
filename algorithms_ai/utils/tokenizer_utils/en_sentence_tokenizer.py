from collections import defaultdict

import re
import types
from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple


def _mro(cls):
    if isinstance(cls, type):
        return cls.__mro__
    else:
        mro = [cls]
        for base in cls.__bases__:
            mro.extend(_mro(base))
        return mro


def overridden(method):
    if isinstance(method, types.MethodType) and method.__self__.__class__ is not None:
        name = method.__name__
        funcs = [
            cls.__dict__[name]
            for cls in _mro(method.__self__.__class__)
            if name in cls.__dict__
        ]
        return len(funcs) > 1
    else:
        raise TypeError("Expected an instance method.")


class TokenizerI(ABC):
    @abstractmethod
    def tokenize(self, s: str) -> List[str]:
        if overridden(self.tokenize_sents):
            return self.tokenize_sents([s])[0]

    def span_tokenize(self, s: str) -> Iterator[Tuple[int, int]]:
        raise NotImplementedError()

    def tokenize_sents(self, strings: List[str]) -> List[List[str]]:
        return [self.tokenize(s) for s in strings]

    def span_tokenize_sents(
            self, strings: List[str]
    ) -> Iterator[List[Tuple[int, int]]]:
        for s in strings:
            yield list(self.span_tokenize(s))


_ORTHO_BEG_UC = 1 << 1
_ORTHO_MID_UC = 1 << 2
_ORTHO_UNK_UC = 1 << 3
_ORTHO_BEG_LC = 1 << 4
_ORTHO_MID_LC = 1 << 5
_ORTHO_UNK_LC = 1 << 6
_ORTHO_UC = _ORTHO_BEG_UC + _ORTHO_MID_UC + _ORTHO_UNK_UC
_ORTHO_LC = _ORTHO_BEG_LC + _ORTHO_MID_LC + _ORTHO_UNK_LC

REASON_DEFAULT_DECISION = "default decision"
REASON_KNOWN_COLLOCATION = "known collocation (both words)"
REASON_ABBR_WITH_ORTHOGRAPHIC_HEURISTIC = "abbreviation + orthographic heuristic"
REASON_ABBR_WITH_SENTENCE_STARTER = "abbreviation + frequent sentence starter"
REASON_INITIAL_WITH_ORTHOGRAPHIC_HEURISTIC = "initial + orthographic heuristic"
REASON_NUMBER_WITH_ORTHOGRAPHIC_HEURISTIC = "initial + orthographic heuristic"
REASON_INITIAL_WITH_SPECIAL_ORTHOGRAPHIC_HEURISTIC = (
    "initial + special orthographic heuristic"
)


class PunktLanguageVars:
    __slots__ = ("_re_period_context", "_re_word_tokenizer")

    def __getstate__(self):
        return 1

    def __setstate__(self, state):
        return 1

    sent_end_chars = (".", "?", "!")
    """Characters which are candidates for sentence boundaries"""

    @property
    def _re_sent_end_chars(self):
        return "[%s]" % re.escape("".join(self.sent_end_chars))

    internal_punctuation = ",:;"  # might want to extend this..
    re_boundary_realignment = re.compile(r'["\')\]}]+?(?:\s+|(?=--)|$)', re.MULTILINE)
    _re_word_start = r"[^\(\"\`{\[:;&\#\*@\)}\]\-,]"

    @property
    def _re_non_word_chars(self):
        return r"(?:[)\";}\]\*:@\'\({\[%s])" % re.escape(
            "".join(set(self.sent_end_chars) - {"."})
        )

    _re_multi_char_punct = r"(?:\-{2,}|\.{2,}|(?:\.\s){2,}\.)"
    """Hyphen and ellipsis are multi-character punctuation"""

    _word_tokenize_fmt = r"""(
        %(MultiChar)s
        |
        (?=%(WordStart)s)\S+?  # Accept word characters until end is found
        (?= # Sequences marking a word's end
            \s|                                 # White-space
            $|                                  # End-of-string
            %(NonWord)s|%(MultiChar)s|          # Punctuation
            ,(?=$|\s|%(NonWord)s|%(MultiChar)s) # Comma if at end of word
        )
        |
        \S
    )"""

    def _word_tokenizer_re(self):
        try:
            return self._re_word_tokenizer
        except AttributeError:
            self._re_word_tokenizer = re.compile(
                self._word_tokenize_fmt
                % {
                    "NonWord": self._re_non_word_chars,
                    "MultiChar": self._re_multi_char_punct,
                    "WordStart": self._re_word_start,
                },
                re.UNICODE | re.VERBOSE,
            )
            return self._re_word_tokenizer

    def word_tokenize(self, s):
        return self._word_tokenizer_re().findall(s)

    _period_context_fmt = r"""
        %(SentEndChars)s             # a potential sentence ending
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            \s+(?P<next_tok>\S+)     # or whitespace and some other token
        ))"""

    def period_context_re(self):
        try:
            return self._re_period_context
        except:
            self._re_period_context = re.compile(
                self._period_context_fmt
                % {
                    "NonWord": self._re_non_word_chars,
                    "SentEndChars": self._re_sent_end_chars,
                },
                re.UNICODE | re.VERBOSE,
            )
            return self._re_period_context


_re_non_punct = re.compile(r"[^\W\d]", re.UNICODE)


def _pair_iter(iterator):
    iterator = iter(iterator)
    try:
        prev = next(iterator)
    except StopIteration:
        return
    for el in iterator:
        yield (prev, el)
        prev = el
    yield (prev, None)


class PunktParameters:
    def __init__(self):
        self.abbrev_types = set()

        self.collocations = set()

        self.sent_starters = set()

        self.ortho_context = defaultdict(int)

    def clear_abbrevs(self):
        self.abbrev_types = set()

    def clear_collocations(self):
        self.collocations = set()

    def clear_sent_starters(self):
        self.sent_starters = set()

    def clear_ortho_context(self):
        self.ortho_context = defaultdict(int)

    def add_ortho_context(self, typ, flag):
        self.ortho_context[typ] |= flag


class PunktToken:
    _properties = ["parastart", "linestart", "sentbreak", "abbr", "ellipsis"]
    __slots__ = ["tok", "type", "period_final"] + _properties

    def __init__(self, tok, **params):
        self.tok = tok
        self.type = self._get_type(tok)
        self.period_final = tok.endswith(".")

        for prop in self._properties:
            setattr(self, prop, None)
        for k in params:
            setattr(self, k, params[k])

    _RE_ELLIPSIS = re.compile(r"\.\.+$")
    _RE_NUMERIC = re.compile(r"^-?[\.,]?\d[\d,\.-]*\.?$")
    _RE_INITIAL = re.compile(r"[^\W\d]\.$", re.UNICODE)
    _RE_ALPHA = re.compile(r"[^\W\d]+$", re.UNICODE)

    def _get_type(self, tok):
        return self._RE_NUMERIC.sub("##number##", tok.lower())

    @property
    def type_no_period(self):
        if len(self.type) > 1 and self.type[-1] == ".":
            return self.type[:-1]
        return self.type

    @property
    def type_no_sentperiod(self):
        if self.sentbreak:
            return self.type_no_period
        return self.type

    @property
    def first_upper(self):
        return self.tok[0].isupper()

    @property
    def first_lower(self):
        return self.tok[0].islower()

    @property
    def first_case(self):
        if self.first_lower:
            return "lower"
        if self.first_upper:
            return "upper"
        return "none"

    @property
    def is_ellipsis(self):
        return self._RE_ELLIPSIS.match(self.tok)

    @property
    def is_number(self):
        return self.type.startswith("##number##")

    @property
    def is_initial(self):
        return self._RE_INITIAL.match(self.tok)

    @property
    def is_alpha(self):
        return self._RE_ALPHA.match(self.tok)

    @property
    def is_non_punct(self):
        return _re_non_punct.search(self.type)

    def __repr__(self):
        typestr = " type=%s," % repr(self.type) if self.type != self.tok else ""

        propvals = ", ".join(
            f"{p}={repr(getattr(self, p))}"
            for p in self._properties
            if getattr(self, p)
        )

        return "{}({},{} {})".format(
            self.__class__.__name__,
            repr(self.tok),
            typestr,
            propvals,
        )

    def __str__(self):
        res = self.tok
        if self.abbr:
            res += "<A>"
        if self.ellipsis:
            res += "<E>"
        if self.sentbreak:
            res += "<S>"
        return res


class PunktBaseClass:
    def __init__(self, lang_vars=None, token_cls=PunktToken, params=None):
        if lang_vars is None:
            lang_vars = PunktLanguageVars()
        if params is None:
            params = PunktParameters()
        self._params = params
        self._lang_vars = lang_vars
        self._Token = token_cls

    def _tokenize_words(self, plaintext):
        parastart = False
        for line in plaintext.split("\n"):
            if line.strip():
                line_toks = iter(self._lang_vars.word_tokenize(line))

                try:
                    tok = next(line_toks)
                except StopIteration:
                    continue

                yield self._Token(tok, parastart=parastart, linestart=True)
                parastart = False

                for tok in line_toks:
                    yield self._Token(tok)
            else:
                parastart = True

    def _annotate_first_pass(self, tokens):
        for aug_tok in tokens:
            self._first_pass_annotation(aug_tok)
            yield aug_tok

    def _first_pass_annotation(self, aug_tok):
        tok = aug_tok.tok

        if tok in self._lang_vars.sent_end_chars:
            aug_tok.sentbreak = True
        elif aug_tok.is_ellipsis:
            aug_tok.ellipsis = True
        elif aug_tok.period_final and not tok.endswith(".."):
            if (
                    tok[:-1].lower() in self._params.abbrev_types
                    or tok[:-1].lower().split("-")[-1] in self._params.abbrev_types
            ):

                aug_tok.abbr = True
            else:
                aug_tok.sentbreak = True

        return


class PunktSentenceTokenizer(PunktBaseClass, TokenizerI):

    def __init__(
            self, params=None, verbose=False, lang_vars=None, token_cls=PunktToken
    ):

        PunktBaseClass.__init__(self, lang_vars=lang_vars, token_cls=token_cls)
        self._params = params

    def tokenize(self, text, realign_boundaries=True):
        return list(self.sentences_from_text(text, realign_boundaries))

    def span_tokenize(self, text, realign_boundaries=True):
        slices = self._slices_from_text(text)
        if realign_boundaries:
            slices = self._realign_boundaries(text, slices)
        for sentence in slices:
            yield (sentence.start, sentence.stop)

    def sentences_from_text(self, text, realign_boundaries=True):
        return [text[s:e] for s, e in self.span_tokenize(text, realign_boundaries)]

    def _match_potential_end_contexts(self, text):
        before_words = {}
        matches = []
        for match in reversed(list(self._lang_vars.period_context_re().finditer(text))):
            if matches and match.end() > before_start:
                continue
            split = text[: match.start()].rsplit(maxsplit=1)
            before_start = len(split[0]) if len(split) == 2 else 0
            before_words[match] = split[-1] if split else ""
            matches.append(match)

        return [
            (
                match,
                before_words[match] + match.group() + match.group("after_tok"),
            )
            for match in matches[::-1]
        ]

    def _slices_from_text(self, text):
        last_break = 0
        for match, context in self._match_potential_end_contexts(text):
            if self.text_contains_sentbreak(context):
                yield slice(last_break, match.end())
                if match.group("next_tok"):
                    last_break = match.start("next_tok")
                else:
                    # next sentence starts at following punctuation
                    last_break = match.end()
        yield slice(last_break, len(text.rstrip()))

    def _realign_boundaries(self, text, slices):
        realign = 0
        for sentence1, sentence2 in _pair_iter(slices):
            sentence1 = slice(sentence1.start + realign, sentence1.stop)
            if not sentence2:
                if text[sentence1]:
                    yield sentence1
                continue

            m = self._lang_vars.re_boundary_realignment.match(text[sentence2])
            if m:
                yield slice(sentence1.start, sentence2.start + len(m.group(0).rstrip()))
                realign = m.end()
            else:
                realign = 0
                if text[sentence1]:
                    yield sentence1

    def text_contains_sentbreak(self, text):
        found = False  # used to ignore last token
        for tok in self._annotate_tokens(self._tokenize_words(text)):
            if found:
                return True
            if tok.sentbreak:
                found = True
        return False

    def _annotate_tokens(self, tokens):
        tokens = self._annotate_first_pass(tokens)
        tokens = self._annotate_second_pass(tokens)
        return tokens

    def _annotate_second_pass(self, tokens):
        for token1, token2 in _pair_iter(tokens):
            self._second_pass_annotation(token1, token2)
            yield token1

    def _second_pass_annotation(self, aug_tok1, aug_tok2):
        if not aug_tok2:
            return

        if not aug_tok1.period_final:
            return
        typ = aug_tok1.type_no_period
        next_typ = aug_tok2.type_no_sentperiod
        tok_is_initial = aug_tok1.is_initial

        if (typ, next_typ) in self._params.collocations:
            aug_tok1.sentbreak = False
            aug_tok1.abbr = True
            return REASON_KNOWN_COLLOCATION

        if (aug_tok1.abbr or aug_tok1.ellipsis) and (not tok_is_initial):
            is_sent_starter = self._ortho_heuristic(aug_tok2)
            if is_sent_starter == True:
                aug_tok1.sentbreak = True
                return REASON_ABBR_WITH_ORTHOGRAPHIC_HEURISTIC

            if aug_tok2.first_upper and next_typ in self._params.sent_starters:
                aug_tok1.sentbreak = True
                return REASON_ABBR_WITH_SENTENCE_STARTER

        if tok_is_initial or typ == "##number##":

            is_sent_starter = self._ortho_heuristic(aug_tok2)

            if is_sent_starter == False:
                aug_tok1.sentbreak = False
                aug_tok1.abbr = True
                if tok_is_initial:
                    return REASON_INITIAL_WITH_ORTHOGRAPHIC_HEURISTIC
                return REASON_NUMBER_WITH_ORTHOGRAPHIC_HEURISTIC

            if (
                    is_sent_starter == "unknown"
                    and tok_is_initial
                    and aug_tok2.first_upper
                    and not (self._params.ortho_context[next_typ] & _ORTHO_LC)
            ):
                aug_tok1.sentbreak = False
                aug_tok1.abbr = True
                return REASON_INITIAL_WITH_SPECIAL_ORTHOGRAPHIC_HEURISTIC

        return

    def _ortho_heuristic(self, aug_tok):
        if aug_tok.tok in (';', ':', ',', '.', '!', '?'):
            return False

        ortho_context = self._params.ortho_context[aug_tok.type_no_sentperiod]

        if (
                aug_tok.first_upper
                and (ortho_context & _ORTHO_LC)
                and not (ortho_context & _ORTHO_MID_UC)
        ):
            return True

        if aug_tok.first_lower and (
                (ortho_context & _ORTHO_UC) or not (ortho_context & _ORTHO_BEG_LC)
        ):
            return False

        return "unknown"


class EnSentenceTokenizer:
    def __init__(self, abbreviation=None, collocations=None, sent_starters=None):
        """
        都小写
        abbreviation 缩略词在分句最后,默认最后一个字符是'.' 但实际不要把'.'写出来，比如一个以点结尾的分句不要让它分句则填写它前面的句子
        collocations 前后要拼接起来的句子，比如结尾JohannS. 和开头Bach，要拼接：{('johanns', 'bach')} # line 476
        sent_starters 已经被标记为句子开头并且真的是句子开头的单词,并且实际还要upper形式  # line 487
        """
        # abbreviation = ['et al', 'i.e', 'e.g', 'etc', 'vs', 'v.s']
        punkt_param = PunktParameters()
        if abbreviation:
            punkt_param.abbrev_types = set(abbreviation)
        if collocations:
            punkt_param.collocations = set(collocations)
        if sent_starters:
            punkt_param.sent_starters = set(sent_starters)
        self.span_sentence_tokenizer = PunktSentenceTokenizer(punkt_param).span_tokenize

    def run(self, text, split_by_new_line=False):
        # split_by_new_line 是否处理换行符号，默认不处理
        # 返回实际字符位置
        if split_by_new_line:
            sentence_index = 0
            sentence_start = 0

            texts = []
            offsets_mapping = []
            texts_append = texts.append
            offsets_mapping_append = offsets_mapping.append

            for each_sentence in re.split('[\n]', text):
                for i, (s, e) in enumerate(self.span_sentence_tokenizer(each_sentence, True)):
                    texts_append(each_sentence[s:e])
                    offsets_mapping_append((s + sentence_start, e - 1 + sentence_start))
                    sentence_index += 1
                sentence_start = sentence_start + len(each_sentence) + 1
            return {'texts': texts, "offsets_mapping": offsets_mapping}
        else:
            texts = []
            offsets_mapping = []
            texts_append = texts.append
            offsets_mapping_append = offsets_mapping.append
            for s, e in self.span_sentence_tokenizer(text, True):
                texts_append(text[s:e]), offsets_mapping_append((s, e - 1))
            return {'texts': texts, "offsets_mapping": offsets_mapping}


if __name__ == '__main__':
    abbreviation = ['et al', 'i.e', 'e.g', 'etc', 'vs', 'v.s', ]
    sent_tokenizer = EnSentenceTokenizer(abbreviation)

    t = '  fight among JohannS. Bach communists and anarchists \n (i.e. at a series of events named May Days).'
    print(sent_tokenizer.run(t, split_by_new_line=True))
