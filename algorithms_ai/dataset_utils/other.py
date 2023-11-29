# 处理实体识别的label
# 输入原文，和标签或预测的实体，输出规范化后的实体
# 原始的label需要包含：start_offset，end_offset，label,以及原文


class FormatLabels:
    def __init__(self,
                 left_token_keep=None,
                 right_token_keep=('%',),
                 left_token_remove=None,
                 right_token_remove=None,
                 left_token_add=None,
                 right_token_add=None
                 ):
        """
        默认对边界('(',')'),('[',']')进行处理，
        根据tokens和labels-token来对label进行规范化
        left_token_remove: label中左边token的去除 , 默认去除所有标点+空格
        right_token_remove: label中右边token的去除，, 默认去除所有标点+空格

        保留的强度大于去除的强度
        left_token_keep：label中左边token的保留,
        right_token_keep：label中右边token的保留

        # 最后在左右两边添加token
        left_token_add =None,
        right_token_add = None

        """

        self.left_remove = list(PUNCTUATIONS) + [' ']
        if left_token_remove:
            for i in left_token_remove:
                self.left_remove.append(i)

        self.right_remove = list(PUNCTUATIONS) + [' ']
        if right_token_remove:
            for i in right_token_remove:
                self.right_remove.append(i)

        if right_token_keep:
            for i in right_token_keep:
                if i in self.right_remove:
                    self.right_remove.remove(i)

        if left_token_keep:
            for i in left_token_keep:
                if i in self.left_remove:
                    self.left_remove.remove(i)

        self.left_add = []
        if left_token_add:
            self.left_add.extend(left_token_add)

        self.right_add = []
        if right_token_add:
            self.right_add.extend(right_token_add)

    def run(self, labels, tokens):
        labels = [self.format_each_label(i, tokens) for i in labels]
        labels = [i for i in labels if i]

        all_labels = [i.label for i in labels]
        refined_labels = []
        for a_l in all_labels:
            # 不同label多个实体的合并
            sub_labels = [i for i in labels if i.label == a_l]
            sub_labels = sorted(sub_labels, key=lambda x: x.start_token.index)
            for i in self.merge_nested_labels(sub_labels):
                if i not in refined_labels:
                    refined_labels.append(i)

        return refined_labels

    def format_each_label(self, label, tokens):
        # ->NER_LABEL
        # 对于label增删左右边界
        while label.start_token.text in self.left_remove:
            label = NER_LABEL(tokens[label.start_token.index + 1], label.end_token, label.label)
            if label.start_token.index > label.end_token.index:  # 全都删完
                return

        while label.end_token.text in self.right_remove:
            label = NER_LABEL(label.start_token, tokens[label.end_token.index - 1], label.label)
            if label.start_token.index > label.end_token.index:  # 全都删完
                return

        all_texts = [i.text for i in tokens[label.start_token.index:label.end_token.index + 1]]

        done_bound = False
        while not done_bound:
            c = 0
            # 补充左右括号
            for left_bound, right_bound in [('(', ')'), ('[', ']')]:
                if all_texts.count(left_bound) - all_texts.count(right_bound) >= 1:
                    if label.end_token.index <= len(tokens) - 2 and tokens[
                        label.end_token.index + 1].text == right_bound:
                        label = NER_LABEL(label.start_token, tokens[label.end_token.index + 1], label.label)
                        break

                if all_texts.count(right_bound) - all_texts.count(left_bound) >= 1:
                    if label.start_token.index >= 1 and tokens[label.start_token.index - 1].text == left_bound:
                        label = NER_LABEL(tokens[label.start_token.index - 1], label.end_token, label.label)
                        break
                c += 1

            if c == 2:
                done_bound = True

        while (label.start_token.index >= 1) and (tokens[label.start_token.index - 1] in self.left_add):
            label = NER_LABEL(tokens[label.start_token.index - 1], label.end_token, label.label)

        while (label.end_token.index + 1 <= len(tokens)) and (tokens[label.end_token.index + 1] in self.right_add):
            label = NER_LABEL(label.start_token, tokens[label.end_token.index + 1], label.label)

        return label

    @staticmethod
    def merge_nested_labels(labels):
        # NER_LABEL
        # 同一种实体的嵌套合并，输入的labels有序 ,交叉的合并，相邻的合并
        # TODO: 两个相邻相同标签的实体，如果中间是’[,],(,)'就把它们合并一起
        refined_labels = []

        last_label = labels[0]
        label = last_label.label
        for current_label in labels[1:]:
            if current_label.start_token.index > last_label.end_token.index + 1:
                refined_labels.append(last_label)
                last_label = current_label
            else:
                last_label = NER_LABEL(
                    last_label.start_token,
                    current_label.end_token,
                    label
                )
        refined_labels.append(last_label)
        return refined_labels

    def get_labels_from_pred_tokens(self, token_predicts):
        # 模型输出各个token的判断结果，将这些结果合并起来,BIO ， I和B有相似地位
        all_labels = []
        append_label = all_labels.append
        label = []
        init_label = label.clear
        append_token = label.append

        last_tag = 'O'
        last_label = 'O'
        for token_index, current_pred in enumerate(token_predicts):
            current_tag = current_pred.split('-')[0]
            current_label = current_pred.split('-')[-1]

            if last_tag == 'O':
                if current_tag in ('B', 'I'):
                    append_token(TOKEN(token_index, -1, -1, '', current_label))
            elif last_tag == 'B':
                if current_tag == 'O':
                    append_label(NER_LABEL(label[0], label[-1], last_label))
                    init_label()
                elif current_tag == 'B':
                    append_label(NER_LABEL(label[0], label[-1], last_label))
                    init_label()
                    append_token(TOKEN(token_index, -1, -1, '', current_label))
                else:  # I
                    if last_label == current_label:
                        append_token(TOKEN(token_index, -1, -1, '', current_label))
                    else:
                        append_label(NER_LABEL(label[0], label[-1], last_label))
                        init_label()
                        append_token(TOKEN(token_index, -1, -1, '', current_label))
            else:  # I
                if current_tag == 'O':
                    append_label(NER_LABEL(label[0], label[-1], last_label))
                    init_label()
                elif current_tag == 'B':  # 另一个实体
                    append_label(NER_LABEL(label[0], label[-1], last_label))
                    init_label()
                    append_token(TOKEN(token_index, -1, -1, '', current_label))
                else:  # 'I'
                    if last_label == current_label:
                        append_token(TOKEN(token_index, -1, -1, '', current_label))
                    else:
                        append_label(NER_LABEL(label[0], label[-1], last_label))
                        init_label()
                        append_token(TOKEN(token_index, -1, -1, '', current_label))
            last_tag = current_tag
            last_label = current_label
        if label:
            append_label(NER_LABEL(label[0], label[-1], last_label))
        return all_labels

    def get_labels_from_pred_tokens_strict(self, token_predicts):
        # 模型输出各个token的判断结果，将这些结果合并起来,BIO ,严格BIO
        all_labels = []
        append_label = all_labels.append
        label = []
        init_label = label.clear
        append_token = label.append

        last_tag = 'O'
        last_label = 'O'
        for token_index, current_pred in enumerate(token_predicts):
            current_tag = current_pred.split('-')[0]
            current_label = current_pred.split('-')[-1]

            if last_tag == 'O':
                if current_tag == 'B':
                    append_token(TOKEN(token_index, -1, -1, '', current_label))
            elif last_tag == 'B':
                if current_tag == 'O':
                    append_label(NER_LABEL(label[0], label[-1], last_label))
                    init_label()
                elif current_tag == 'B':
                    append_label(NER_LABEL(label[0], label[-1], last_label))
                    init_label()
                    append_token(TOKEN(token_index, -1, -1, '', current_label))
                else:  # I
                    if last_label == current_label:
                        append_token(TOKEN(token_index, -1, -1, '', current_label))
            else:  # I
                if current_tag == 'O':
                    if label:
                        append_label(NER_LABEL(label[0], label[-1], last_label))
                        init_label()
                elif current_tag == 'B':  # 另一个实体
                    if label:
                        append_label(NER_LABEL(label[0], label[-1], last_label))
                    init_label()
                    append_token(TOKEN(token_index, -1, -1, '', current_label))
                else:  # 'I'
                    if last_label == current_label:
                        if label:
                            append_token(TOKEN(token_index, -1, -1, '', current_label))
            last_tag = current_tag
            last_label = current_label
        if label:
            append_label(NER_LABEL(label[0], label[-1], last_label))

        return all_labels

    def simple_process_input_text(self, input_text):
        # 规范化处理原始文本，得到处理后的文本，文本的tokens和处理后文本的tokens
        tokens = self.word_tokenizer(input_text)
        refined_input_text = ''
        last_end = -1

        for token in tokens:
            token_text = self.token_processor(token.text) if self.token_processor(token.text) else ' '
            if token.start_offset == last_end + 1:
                refined_input_text += token_text
            else:
                refined_input_text += (' ' + token_text)
            last_end = token.end_offset

        return refined_input_text.strip()


# class FormatLabels:
#     def __init__(self,
#                  left_token_keep=None,
#                  right_token_keep=('%',),
#                  left_token_remove=None,
#                  right_token_remove=None,
#                  left_token_add=None,
#                  right_token_add=None
#                  ):
#         """
#         默认对边界('(',')'),('[',']')进行处理，
#         根据tokens和labels-token来对label进行规范化
#         left_token_remove: label中左边token的去除 , 默认去除所有标点+空格
#         right_token_remove: label中右边token的去除，, 默认去除所有标点+空格
#
#         保留的强度大于去除的强度
#         left_token_keep：label中左边token的保留,
#         right_token_keep：label中右边token的保留
#
#         # 最后在左右两边添加token
#         left_token_add = None
#         right_token_add = None
#
#         """
#
#         self.left_remove = list(PUNCTUATIONS) + [' ']
#         if left_token_remove:
#             for i in left_token_remove:
#                 self.left_remove.append(i)
#
#         self.right_remove = list(PUNCTUATIONS) + [' ']
#         if right_token_remove:
#             for i in right_token_remove:
#                 self.right_remove.append(i)
#
#         if right_token_keep:
#             for i in right_token_keep:
#                 if i in self.right_remove:
#                     self.right_remove.remove(i)
#
#         if left_token_keep:
#             for i in left_token_keep:
#                 if i in self.left_remove:
#                     self.left_remove.remove(i)
#
#         self.left_add = []
#         if left_token_add:
#             self.left_add.extend(left_token_add)
#
#         self.right_add = []
#         if right_token_add:
#             self.right_add.extend(right_token_add)
#
#     def run(self, labels, tokens):
#         labels = [self.format_each_label(i, tokens) for i in labels]
#         labels = [i for i in labels if i]
#
#         all_labels = [i.label for i in labels]
#         refined_labels = []
#         for a_l in all_labels:
#             # 不同label多个实体的合并
#             sub_labels = [i for i in labels if i.label == a_l]
#             sub_labels = sorted(sub_labels, key=lambda x: x.start_token.index)
#             for i in self.merge_nested_labels(sub_labels):
#                 if i not in refined_labels:
#                     refined_labels.append(i)
#
#         return refined_labels
#
#     def format_each_label(self, label, tokens):
#         # ->TokenLabel
#         # 对于label增删左右边界
#         while label.start_token.text in self.left_remove:
#             label = TokenLabel(tokens[label.start_token.index + 1], label.end_token, label.label)
#             if label.start_token.index > label.end_token.index:  # 全都删完
#                 return
#
#         while label.end_token.text in self.right_remove:
#             label = TokenLabel(label.start_token, tokens[label.end_token.index - 1], label.label)
#             if label.start_token.index > label.end_token.index:  # 全都删完
#                 return
#
#         all_texts = [i.text for i in tokens[label.start_token.index:label.end_token.index + 1]]
#
#         done_bound = False
#         while not done_bound:
#             c = 0
#             # 补充左右括号
#             for left_bound, right_bound in [('(', ')'), ('[', ']')]:
#                 if all_texts.count(left_bound) - all_texts.count(right_bound) >= 1:
#                     if label.end_token.index <= len(tokens) - 2 and tokens[
#                         label.end_token.index + 1].text == right_bound:
#                         label = TokenLabel(label.start_token, tokens[label.end_token.index + 1], label.label)
#                         break
#
#                 if all_texts.count(right_bound) - all_texts.count(left_bound) >= 1:
#                     if label.start_token.index >= 1 and tokens[label.start_token.index - 1].text == left_bound:
#                         label = TokenLabel(tokens[label.start_token.index - 1], label.end_token, label.label)
#                         break
#                 c += 1
#
#             if c == 2:
#                 done_bound = True
#
#         while (label.start_token.index >= 1) and (tokens[label.start_token.index - 1] in self.left_add):
#             label = TokenLabel(tokens[label.start_token.index - 1], label.end_token, label.label)
#
#         while (label.end_token.index + 1 <= len(tokens)) and (tokens[label.end_token.index + 1] in self.right_add):
#             label = TokenLabel(label.start_token, tokens[label.end_token.index + 1], label.label)
#
#         return label
#
#     @staticmethod
#     def merge_nested_labels(labels):
#         # TokenLabel
#         # 同一种实体的嵌套合并，输入的labels有序 ,交叉的合并，相邻的合并
#         # TODO: 两个相邻相同标签的实体，如果中间是’[,],(,)'就把它们合并一起
#         refined_labels = []
#
#         last_label = labels[0]
#         label = last_label.label
#         for current_label in labels[1:]:
#             if current_label.start_token.index > last_label.end_token.index + 1:
#                 refined_labels.append(last_label)
#                 last_label = current_label
#             else:
#                 last_label = TokenLabel(
#                     last_label.start_token,
#                     current_label.end_token,
#                     label
#                 )
#         refined_labels.append(last_label)
#         return refined_labels
#
#     def get_labels_from_pred_tokens(self, token_predicts):
#         # 模型输出各个token的判断结果，将这些结果合并起来,BIO ， I和B有相似地位
#         all_labels = []
#         append_label = all_labels.append
#         label = []
#         init_label = label.clear
#         append_token = label.append
#
#         last_tag = 'O'
#         last_label = 'O'
#         for token_index, current_pred in enumerate(token_predicts):
#             current_tag = current_pred.split('-')[0]
#             current_label = current_pred.split('-')[-1]
#
#             if last_tag == 'O':
#                 if current_tag in ('B', 'I'):
#                     append_token(Token(token_index, -1, -1, '', current_label))
#             elif last_tag == 'B':
#                 if current_tag == 'O':
#                     append_label(TokenLabel(label[0], label[-1], last_label))
#                     init_label()
#                 elif current_tag == 'B':
#                     append_label(TokenLabel(label[0], label[-1], last_label))
#                     init_label()
#                     append_token(Token(token_index, -1, -1, '', current_label))
#                 else:  # I
#                     if last_label == current_label:
#                         append_token(Token(token_index, -1, -1, '', current_label))
#                     else:
#                         append_label(TokenLabel(label[0], label[-1], last_label))
#                         init_label()
#                         append_token(Token(token_index, -1, -1, '', current_label))
#             else:  # I
#                 if current_tag == 'O':
#                     append_label(TokenLabel(label[0], label[-1], last_label))
#                     init_label()
#                 elif current_tag == 'B':  # 另一个实体
#                     append_label(TokenLabel(label[0], label[-1], last_label))
#                     init_label()
#                     append_token(Token(token_index, -1, -1, '', current_label))
#                 else:  # 'I'
#                     if last_label == current_label:
#                         append_token(Token(token_index, -1, -1, '', current_label))
#                     else:
#                         append_label(TokenLabel(label[0], label[-1], last_label))
#                         init_label()
#                         append_token(Token(token_index, -1, -1, '', current_label))
#             last_tag = current_tag
#             last_label = current_label
#         if label:
#             append_label(TokenLabel(label[0], label[-1], last_label))
#         return all_labels
#
#     def get_labels_from_pred_tokens_strict(self, token_predicts):
#         # 模型输出各个token的判断结果，将这些结果合并起来,BIO ,严格BIO
#         all_labels = []
#         append_label = all_labels.append
#         label = []
#         init_label = label.clear
#         append_token = label.append
#
#         last_tag = 'O'
#         last_label = 'O'
#         for token_index, current_pred in enumerate(token_predicts):
#             current_tag = current_pred.split('-')[0]
#             current_label = current_pred.split('-')[-1]
#
#             if last_tag == 'O':
#                 if current_tag == 'B':
#                     append_token(Token(token_index, -1, -1, '', current_label))
#             elif last_tag == 'B':
#                 if current_tag == 'O':
#                     append_label(TokenLabel(label[0], label[-1], last_label))
#                     init_label()
#                 elif current_tag == 'B':
#                     append_label(TokenLabel(label[0], label[-1], last_label))
#                     init_label()
#                     append_token(Token(token_index, -1, -1, '', current_label))
#                 else:  # I
#                     if last_label == current_label:
#                         append_token(Token(token_index, -1, -1, '', current_label))
#             else:  # I
#                 if current_tag == 'O':
#                     if label:
#                         append_label(TokenLabel(label[0], label[-1], last_label))
#                         init_label()
#                 elif current_tag == 'B':  # 另一个实体
#                     if label:
#                         append_label(TokenLabel(label[0], label[-1], last_label))
#                     init_label()
#                     append_token(Token(token_index, -1, -1, '', current_label))
#                 else:  # 'I'
#                     if last_label == current_label:
#                         if label:
#                             append_token(Token(token_index, -1, -1, '', current_label))
#             last_tag = current_tag
#             last_label = current_label
#         if label:
#             append_label(TokenLabel(label[0], label[-1], last_label))
#
#         return all_labels


# supplement label

# class LabelProcessor:
#     def __init__(self,
#                  left_token_keep=None,
#                  right_token_keep=('%',),
#                  left_token_remove=None,
#                  right_token_remove=None,
#                  left_token_add=None,
#                  right_token_add=None
#                  ):
#         """
#         默认对边界('(',')'),('[',']')进行处理，
#         根据tokens和labels-token来对label进行规范化
#         left_token_remove: label中左边token的去除 , 默认去除所有标点+空格
#         right_token_remove: label中右边token的去除，, 默认去除所有标点+空格
#
#         保留的强度大于去除的强度
#         left_token_keep：label中左边token的保留,
#         right_token_keep：label中右边token的保留
#
#         # 最后在左右两边添加token
#         left_token_add =None,
#         right_token_add = None
#
#         """
#
#         self.left_remove = list(PUNCTUATIONS) + [' ']
#         if left_token_remove:
#             for i in left_token_remove:
#                 self.left_remove.append(i)
#
#         self.right_remove = list(PUNCTUATIONS) + [' ']
#         if right_token_remove:
#             for i in right_token_remove:
#                 self.right_remove.append(i)
#
#         if right_token_keep:
#             for i in right_token_keep:
#                 if i in self.right_remove:
#                     self.right_remove.remove(i)
#
#         if left_token_keep:
#             for i in left_token_keep:
#                 if i in self.left_remove:
#                     self.left_remove.remove(i)
#
#         self.left_add = []
#         if left_token_add:
#             self.left_add.extend(left_token_add)
#
#         self.right_add = []
#         if right_token_add:
#             self.right_add.extend(right_token_add)
#
#     def run(self, labels, tokens):
#         labels = [self.format_each_label(i, tokens) for i in labels]
#         labels = [i for i in labels if i]
#
#         all_labels = [i.label for i in labels]
#         refined_labels = []
#         for a_l in all_labels:
#             # 不同label多个实体的合并
#             sub_labels = [i for i in labels if i.label == a_l]
#             sub_labels = sorted(sub_labels, key=lambda x: x.start_token.index)
#             for i in self.merge_nested_labels(sub_labels):
#                 if i not in refined_labels:
#                     refined_labels.append(i)
#
#         return refined_labels
#
#     def format_each_label(self, label, tokens):
#         # ->TokenLabel
#         # 对于label增删左右边界
#         while label.start_token.text in self.left_remove:
#             label = TokenLabel(tokens[label.start_token.index + 1], label.end_token, label.label)
#             if label.start_token.index > label.end_token.index:  # 全都删完
#                 return
#
#         while label.end_token.text in self.right_remove:
#             label = TokenLabel(label.start_token, tokens[label.end_token.index - 1], label.label)
#             if label.start_token.index > label.end_token.index: # 全都删完
#                 return
#
#         all_texts = [i.text for i in tokens[label.start_token.index:label.end_token.index + 1]]
#
#         done_bound = False
#         while not done_bound:
#             c = 0
#             # 补充左右括号
#             for left_bound, right_bound in [('(', ')'), ('[', ']')]:
#                 if all_texts.count(left_bound) - all_texts.count(right_bound) >= 1:
#                     if label.end_token.index  <= len(tokens) - 2 and tokens[label.end_token.index + 1].text == right_bound:
#                         label = TokenLabel(label.start_token, tokens[label.end_token.index + 1], label.label)
#                         break
#
#                 if all_texts.count(right_bound) - all_texts.count(left_bound) >= 1:
#                     if label.start_token.index >= 1 and tokens[label.start_token.index - 1].text == left_bound:
#                         label = TokenLabel(tokens[label.start_token.index - 1], label.end_token, label.label)
#                         break
#                 c += 1
#
#             if c == 2:
#                 done_bound = True
#
#         while (label.start_token.index >= 1) and (tokens[label.start_token.index - 1] in self.left_add):
#             label = TokenLabel(tokens[label.start_token.index - 1], label.end_token, label.label)
#
#         while (label.end_token.index + 1 <= len(tokens)) and (tokens[label.end_token.index + 1] in self.right_add):
#             label = TokenLabel(label.start_token, tokens[label.end_token.index + 1], label.label)
#
#         return label
#
#     @staticmethod
#     def merge_nested_labels(labels):
#         # TokenLabel
#         # 同一种实体的嵌套合并，输入的labels有序 ,交叉的合并，相邻的合并
#         # TODO: 两个相邻相同标签的实体，如果中间是’[,],(,)'就把它们合并一起
#         refined_labels = []
#
#         last_label = labels[0]
#         label = last_label.label
#         for current_label in labels[1:]:
#             if current_label.start_token.index > last_label.end_token.index+1:
#                 refined_labels.append(last_label)
#                 last_label = current_label
#             else:
#                 last_label = TokenLabel(
#                     last_label.start_token,
#                     current_label.end_token,
#                     label
#                 )
#         refined_labels.append(last_label)
#         return refined_labels
#
#     def get_labels_from_pred_tokens(self, token_predicts):
#         # 模型输出各个token的判断结果，将这些结果合并起来,BIO ， I和B有相似地位
#         all_labels = []
#         append_label = all_labels.append
#         label = []
#         init_label = label.clear
#         append_token = label.append
#
#         last_tag = 'O'
#         last_label = 'O'
#         for token_index, current_pred in enumerate(token_predicts):
#             current_tag = current_pred.split('-')[0]
#             current_label = current_pred.split('-')[-1]
#
#             if last_tag == 'O':
#                 if current_tag in ('B', 'I'):
#                     append_token(Token(token_index, -1, -1, '', current_label))
#             elif last_tag == 'B':
#                 if current_tag == 'O':
#                     append_label(TokenLabel(label[0], label[-1], last_label))
#                     init_label()
#                 elif current_tag == 'B':
#                     append_label(TokenLabel(label[0], label[-1], last_label))
#                     init_label()
#                     append_token(Token(token_index, -1, -1, '', current_label))
#                 else:  # I
#                     if last_label == current_label:
#                         append_token(Token(token_index, -1, -1, '', current_label))
#                     else:
#                         append_label(TokenLabel(label[0], label[-1], last_label))
#                         init_label()
#                         append_token(Token(token_index, -1, -1, '', current_label))
#             else:  # I
#                 if current_tag == 'O':
#                     append_label(TokenLabel(label[0], label[-1], last_label))
#                     init_label()
#                 elif current_tag == 'B':  # 另一个实体
#                     append_label(TokenLabel(label[0], label[-1], last_label))
#                     init_label()
#                     append_token(Token(token_index, -1, -1, '', current_label))
#                 else:  # 'I'
#                     if last_label == current_label:
#                         append_token(Token(token_index, -1, -1, '', current_label))
#                     else:
#                         append_label(TokenLabel(label[0], label[-1], last_label))
#                         init_label()
#                         append_token(Token(token_index, -1, -1, '', current_label))
#             last_tag = current_tag
#             last_label = current_label
#         if label:
#             append_label(TokenLabel(label[0], label[-1], last_label))
#         return all_labels
#
#     def get_labels_from_pred_tokens_strict(self, token_predicts):
#         # 模型输出各个token的判断结果，将这些结果合并起来,BIO ,严格BIO
#         all_labels = []
#         append_label = all_labels.append
#         label = []
#         init_label = label.clear
#         append_token = label.append
#
#         last_tag = 'O'
#         last_label = 'O'
#         for token_index, current_pred in enumerate(token_predicts):
#             current_tag = current_pred.split('-')[0]
#             current_label = current_pred.split('-')[-1]
#
#             if last_tag == 'O':
#                 if current_tag == 'B':
#                     append_token(Token(token_index, -1, -1, '', current_label))
#             elif last_tag == 'B':
#                 if current_tag == 'O':
#                     append_label(TokenLabel(label[0], label[-1], last_label))
#                     init_label()
#                 elif current_tag == 'B':
#                     append_label(TokenLabel(label[0], label[-1], last_label))
#                     init_label()
#                     append_token(Token(token_index, -1, -1, '', current_label))
#                 else:  # I
#                     if last_label == current_label:
#                         append_token(Token(token_index, -1, -1, '', current_label))
#             else:  # I
#                 if current_tag == 'O':
#                     if label:
#                         append_label(TokenLabel(label[0], label[-1], last_label))
#                         init_label()
#                 elif current_tag == 'B':  # 另一个实体
#                     if label:
#                         append_label(TokenLabel(label[0], label[-1], last_label))
#                     init_label()
#                     append_token(Token(token_index, -1, -1, '', current_label))
#                 else:  # 'I'
#                     if last_label == current_label:
#                         if label:
#                             append_token(Token(token_index, -1, -1, '', current_label))
#             last_tag = current_tag
#             last_label = current_label
#         if label:
#             append_label(TokenLabel(label[0], label[-1], last_label))
#
#         return all_labels


"""
{
  "文本处理": [
    "token切分器,WordTokenizer-tokenize_text",
    "token处理器,TokenProcessor - process_token_text",
    "文本处理把原始文本处理成token级别的文本，以及token的映射，然后对原始的标签进行token的映射，得到label的token表示，refined_label"
  ],
  "中间": "对文本处理得到文本token映射和refined-token,若有label，可以把label规范化为list(token)格式",
  "多种实体分实体进行处理":{
    "输入输出": "输入的是input_text和label，中间refined_text 输出refined_label",
    "标签处理": [
    "标签的修正，主要边界的处理，左右边界的增删改查，和中间token的切分",
    "标签的删除，主要删除错误的标签--判断--输入label和refined_text,得出true/false",
    "标签的补充，主要使用规则抽取实体--，一般使用re规则，使用refined_input_text，通过正则匹配得出的结果还要映射成token形式",
    "标签的融合，一般相邻或交叉变成一个,--输入多个实体"
  ],
"多实体处理": "融合多实体，或保留",
"desc": "通过处理refined_input_text和label，得到 refined_label"},
  "最后": "可以通过input_text,和refined_label得到原始label；",
  "另外": "如果加入模型，输入原始文本，处理成refined_input_text，进入模型，输出refined_label,再进入标签处理，最后将处理后的标签输出",
  "所以": "有四个变量:input_text,refined_input_text,label,refined_label,训练或进入模型用refined_input_text和refined_label，输出给用户是input_text和label",
  "其他": "中间涉及到的变量，原始和结果文本token的映射：tokens，mapping_tokens",
  "注意": "要添加log和debug信息"
}


"""


"""
class TmpToken:
    def __init__(self, pred='O', token_text='[init_token]', tmp_start=0, text='xxxx'):
        self.entity_type_tag = pred[0]
        self.entity_type = pred[2:]
        self.start_offset, self.end_offset = self._get_local(token_text, text, tmp_start)  # 坐标

    def _get_local(self, token_text, raw_text, tmp_start):
        tmp_start = tmp_start + 1
        sub_text = raw_text[tmp_start:]
        if token_text.startswith('##') and len(token_text) > 2:  # token是切分的
            token_text = token_text[2:]
        token_text_index = sub_text.find(token_text.lower())

        if token_text_index == -1:  # token不在原文中
            return 'not_found', 'not_found'
        else:  # 正常token
            return token_text_index + tmp_start, token_text_index + tmp_start + len(token_text) - 1

    def __str__(self):
        return str({
            'start_offset': self.start_offset,
            'end_offset': self.end_offset,
            'entity_type': self.entity_type,
            'entity_type_tag': self.entity_type_tag
        })


def refined_entity(entity_tokens: list, text: str):
    # 规范化token实体
    return {"start_offset": (start_offset := max(0, entity_tokens[0].start_offset)),
            "end_offset": (end_offset := min(entity_tokens[-1].end_offset, len(text) - 1)),  # 坐标
            "text": text[start_offset:end_offset + 1],
            "label": entity_tokens[0].entity_type}


# 将预测的logist值映射为label格式,处理的文本是预处理后的输入模型的文本，不是原始文本
def align_labels_by_token(preds_list, tokens_list, preprocessed_text):
    text = preprocessed_text.lower()
    tmp_end_offset = -1
    last_token = TmpToken('O', '[CLS]', tmp_end_offset, text)
    entities = []  # 存储所有实体的实体token
    append_entity = entities.append
    entity = []  # 存储单个实体的所有token
    init_entity = entity.clear
    append_token = entity.append

    # try:
    for token, pred in zip(tokens_list[1:-1], preds_list[1:-1]):
        current_token = TmpToken(pred, token, tmp_end_offset, text)
        if current_token.start_offset == 'not_found':  # 跳过错误的token
            continue
        if last_token.entity_type_tag == 'O':
            if current_token.entity_type_tag == 'B':
                append_token(current_token)
            elif current_token.entity_type_tag == 'I':
                append_token(current_token)
        elif last_token.entity_type_tag == 'B':
            if current_token.entity_type_tag == 'O':
                append_entity(refined_entity(entity, preprocessed_text))
                init_entity()
            elif current_token.entity_type_tag == 'B':
                if last_token.entity_type == current_token.entity_type:
                    append_token(current_token)
                else:
                    append_entity(refined_entity(entity, preprocessed_text))
                    init_entity()
                    append_token(current_token)
            else:  # 'I'
                if current_token.entity_type == last_token.entity_type:
                    append_token(current_token)
                else:
                    append_entity(refined_entity(entity, preprocessed_text))
                    init_entity()
                    append_token(current_token)
        else:  # 'I'
            if current_token.entity_type_tag == 'O':
                append_entity(refined_entity(entity, preprocessed_text))
                init_entity()
            elif current_token.entity_type_tag == 'B':  # 另一个实体
                if current_token.entity_type == last_token.entity_type:
                    append_token(current_token)
                else:
                    append_entity(refined_entity(entity, preprocessed_text))
                    init_entity()
                    append_token(current_token)
            else:  # 'I'
                if current_token.entity_type == last_token.entity_type:
                    append_token(current_token)
                else:
                    append_entity(refined_entity(entity, preprocessed_text))
                    init_entity()
                    append_token(current_token)
        last_token = current_token
        tmp_end_offset = current_token.end_offset
    if entity:
        append_entity(refined_entity(entity, preprocessed_text))
    return entities

"""

if __name__ == '__main__':
    # s = 'With median follow-up of 21 months, 24-month relapse-free survival (RFS) was 67% (95% CI 62% to 73%) in the 326 patients.'
    # NerFormat().run(s,[])
    s2 = "Oral acalabruti(nib) 100 mg 我i在发啊 <  twic\u00b7e daily  ; was & admi\nnister\ted with or &lt; 13.8kg/m without,α-1 12.😊 "
    labels = [{'start_offset': 14, 'end_offset': 18, 'text': 'i(nib', 'label': 'a'},  # end_offset错误
              # {'start_offset': 16, 'end_offset': 20, 'text': 'nib)', 'label': 'a'},  # 正确
              {'start_offset': 38, 'end_offset': 44, 'text': 'wice dail', 'label': 'c'},  # 边界不对
              {'start_offset': 46, 'end_offset': 54, 'text': 'ly  ; ', 'label': 'c'},
              {'start_offset': 79, 'end_offset': 89, 'text': '&lt; 13.8', 'label': 'd'}]
    s = [{'start_offset': 5, 'end_offset': 18, 'text': 'acalabruti(nib', 'label': 'a'},
         {'start_offset': 38, 'end_offset': 54, 'text': 'twice daily ; was', 'label': 'c'},
         {'start_offset': 78, 'end_offset': 87, 'text': 'or < 13.8', 'label': 'd'}]

    # s2 = "sd fatigue (62 [55.4%]) and gynecomastia (41 [36.6%])sdf "
    # s = {'id': '18612151',
    #      'arms_ner': [{'start_offset': 173,
    #                    'end_offset': 216,
    #                    'text': '(paclitaxel plus carboplatin) with cetuximab',
    #                    'label': 'arm_option'}],
    #      'input_text': 'Two hundred twenty-nine chemotherapy-naive patients with advanced-stage NSCLC were enrolled onto a phase II selection trial evaluating sequential or concurrent chemotherapy (paclitaxel plus carboplatin) with cetuximab.'}
    # labels = [{"start_offset": 3, "end_offset": 125-77+3, "text": "fatigue (62 [55.4%]) and gynecomastia (41 [36.6%]", "label": "result"}]
    print(NER_Processor().process_raw_data(s2, labels))
    print(NER_Processor().text_processor.simple_process_input_text(s2))
    print(s2[78:87])
    print(NER_Processor().get_raw_labels(s2, s))
    k = 'Oral acalabruti(nib) 100 mg 我i在发啊 < twic·e daily ; was & admi nister ed with or < 13.8kg/m without,α-1 12.😊'
    print(k[78:87])
