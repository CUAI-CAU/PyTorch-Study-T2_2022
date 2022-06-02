import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        '''
        e = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)

        Token Embedding : Glove, Word2vec, Fasttext와 같은 것을 사용해 Vector 값을 가져온다.

        Segment Embedding : 단어가 첫번째 문장에 속하는지 두번째 문장에 속하는지 알려준다.

        [ 0 0 0 0 0 1 1 1 1 ] 과 같이 표현하며 해당 Vector를 Token Embedding 차원수와 같게 맞추어 임베딩 해준다.

        Positional Embedding : 각 단어가 첫번째인지 두번째인지를 의미하는 Embedding 값이다. 마찬가지로 Token Embedding의 차원수와 맟춰 임베딩 한 후 더해준다.
        '''
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)
