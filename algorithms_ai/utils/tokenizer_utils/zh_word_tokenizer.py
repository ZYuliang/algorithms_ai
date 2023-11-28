

def cut_sentence_to_words_zh(sentence: str):
    """
        cut_sentence_to_words_zh
    Args:
        sentence: a sentence ,str

    Returns:
        sentences: list[str]
    """
    english = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789αγβδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'
    output = []
    buffer = ''
    for s in sentence:
        if s in english or s in english.upper():  # 英文或数字
            buffer += s
        else:  # 中文
            if buffer:
                output.append(buffer)
            buffer = ''
            output.append(s)
    if buffer:
        output.append(buffer)
    return output
