import numpy as np
from utils.snippets import sequence_padding, DataGenerator
from utils.snippets import open, ViterbiDecoder, to_array


def load_data(filename, categories):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    with open(filename, encoding = 'utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            # print(l)
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                char, flag = c.split('\t')
                d[0] += char
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                    categories.add(flag[2:])
                elif flag[0] == 'I':
                    # print(d) # 在此报错是因为BIO的缘故！
                    d[-1][1] = i
            D.append(d)
    return D


class data_generator(DataGenerator):
    """数据生成器
    """
    
    def __init__(self, data, batch_size, tokenizer, categories, maxlen):
        super().__init__(data = data, batch_size = batch_size)
        self.tokenizer = tokenizer
        self.categories = categories
        self.maxlen = maxlen
    
    def __iter__(self, random = False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = self.tokenizer.tokenize(d[0], maxlen = self.maxlen)
            mapping = self.tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = self.tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros(len(token_ids))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[start] = self.categories.index(label) * 2 + 1
                    labels[start + 1:end + 1] = self.categories.index(label) * 2 + 2
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    
    def __init__(self, tokenizer, model, categories, trans, starts, ends):
        self.tokenizer = tokenizer
        self.model = model
        self.categories = categories
        super().__init__(trans = trans, starts = starts, ends = ends)
    
    def recognize(self, text):
        tokens = self.tokenizer.tokenize(text, maxlen = 512)
        mapping = self.tokenizer.rematch(text, tokens)
        token_ids = self.tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = self.model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], self.categories[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        return [(mapping[w[0]][0], mapping[w[-1]][-1], l) for w, l in entities]
