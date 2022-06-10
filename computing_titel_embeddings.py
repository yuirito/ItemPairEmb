# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This basic example loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of sentences.
"""

import sys

sys.path.append('..')
from text2vec import SentenceModel, EncoderType
from text2vec import Word2Vec


def compute_emb(model):
    # Embed a list of sentences
    
    sentences = [
        '沃猫商用火锅电磁炉800W瓦嵌入式一人一锅自助专用单人旋转火锅',
        '雀巢咖啡速溶咖啡粉1+2原味15克x100条整盒微研磨三合一速溶提神',
        '如何更换花呗绑定银行卡',
        '花呗更改绑定银行卡',
        'This framework generates embeddings for each input sentence',
        'Sentences are passed as a list of string.',
        'The quick brown fox jumps over the lazy dog.'
    ]
    sentence_embeddings = model.encode(sentences, show_progress_bar=True)
    print(type(sentence_embeddings), sentence_embeddings.shape)

    # The result is a list of sentence embeddings as numpy arrays
    for sentence, embedding in zip(sentences, sentence_embeddings):
        print("Sentence:", sentence)
        print("Embedding shape:", embedding.shape)
        print("Embedding head:", embedding[:10])
        print()


if __name__ == "__main__":
    # 中文句向量模型(CoSENT)，中文语义匹配任务推荐，支持fine-tune继续训练
    t2v_model = SentenceModel("shibing624/text2vec-base-chinese",
                              encoder_type=EncoderType.FIRST_LAST_AVG)
    compute_emb(t2v_model)

    
