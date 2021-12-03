# Based on https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5
# https://github.com/jhlau/doc2vec

from gensim.models.doc2vec import Doc2Vec
from nltk import word_tokenize

### Example for Doc2vec Inference
# model= Doc2Vec.load("d2v.model")
# #to find the vector of a document which is not in training data
# test_data = word_tokenize("I love chatbots".lower())
# v1 = model.infer_vector(test_data)
# print(("V1 infer, ", v1))
# print(v1.shape)
#
# # to find most similar doc using tags
# similar_doc = model.dv.most_similar('1')
# print(similar_doc)
#
# # to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
# print(model.dv['1'])


def doc2vec_get_vector(text, model):
    text_tokenized = word_tokenize(text.lower())
    v1 = model.infer_vector(text_tokenized)
    return v1

if __name__ == '__main__':
    model_path = "doc2vec_small.model"
    model = Doc2Vec.load(model_path)
    sentence = "This is a test paper."

    v = doc2vec_get_vector(text=sentence, model=model)
    print(sentence, " ->\n ", v)
