# Based on https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5
# https://github.com/jhlau/doc2vec

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk


def my_train_doc2vec(train_data):
    # nltk.download('punkt')
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

    max_epochs = 100
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002

        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("doc2vec_small.model")
    print("Model Saved")



data = ["I love machine learning. Its awesome.",
        "I love coding in python",
        "I love building chatbots",
        "they chat amagingly well"]

my_train_doc2vec(train_data=data)
