import os
import gensim
from gensim.models import Word2Vec


def clean_fname(fname):
    tmp = fname.split(' ')[0][:-1].lower()
    if fname.count(',') > 1:
        tmp = tmp + "_etal"
    return tmp
    # tmp = fname.split('.')[0]
    # tmp = fname.split(' ')[0][:-1].lower()
    # if fname.count(',') > 1:
    #     tmp = tmp + "_etal"
    # return tmp


fields = ['author']
odir = '/dsl/eebo/wordembeddings/2020.03.02'

for field in fields:
    model_dir = odir

    if field == 'full':
        model_dir += "/"
    else:
        model_dir += '-' + field + 's/'

    print("model directory: " + model_dir)
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
            print("file: " + f)
        else:
            f = f[:-6]  # remove '.model' extension

        ofile = open(model_dir + "results/" + f + ".freq", "w")
        model = Word2Vec.load(path)
        total = sum([model.wv.vocab[word].count for word, vocab_obj in model.wv.vocab.items()])
        for word, vocab_obj in model.wv.vocab.items():
            count = model.wv.vocab[word].count
            freq = count / total
            print(word + " " + str(count) + " " + str(freq), file=ofile)
