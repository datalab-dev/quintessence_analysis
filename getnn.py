import os
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models import translation_matrix
from gensim.models import KeyedVectors


def clean_fname(fname):
    tmp = fname.split(' ')[0][:-1].lower()
    if fname.count(',') > 1:
        tmp = tmp + "_etal"
    return tmp


fields = ['full', 'decade', 'location', 'author']
odir = '/dsl/eebo/wordembeddings/2020.03.02'

for field in fields:
    model_dir = odir

    if field == 'full':
        model_dir += "/"
    else:
        model_dir += '-' + field + 's/'

    print(model_dir)
    files = os.listdir(model_dir)
    files.sort()

    fnames = []
    for f in files:
        if f[-6:] == ".model":
            fnames.append(f)

    for f in fnames:
        path = model_dir + f
        print(path)

        if field == 'author':
            f = clean_fname(f)
        else:
            f = f[:-6]  # remove '.model' extension

        nnofile = open(model_dir + "results/" + f + ".neighbors", "w")

        model = Word2Vec.load(path)
        wv = model.wv
        for k in wv.vocab.keys():
            word = k
            nn = []
            results = wv.most_similar(word, topn=20)
            for m in results:
               nn.append(m[0] + " " + str(m[1]))
            print(word + " " + " ".join(nn), file=nnofile)
