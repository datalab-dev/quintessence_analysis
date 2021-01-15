""" TODO """
def align(self):
    """
    Create terms.timeseries.
    """
    # TODO add smart_procrustes_align_gensim (maybe in utils?)

    # TODO load models  + model names from somwhere
    # TODO t0 = model for 1700

    t0.init_sims()

    dists = {}
    for model in models:
        model.init_sims()
        aligned = smart_procrustes_align_gensim(t0, model)
        for term in aligned.wv.vocab.keys():
            base = t0.wv[term]
            vec = aligned.wv[term]
            similarity = 1 - scipy.spatial.distance.cosine(base, vec)
            if v not in dists:
                dists[term] = np.zeros(len(models), dtype=float)
            dists[term][i] = similarity

    docs = [{'_id': term, 'timeseries': dists[term]} for term in dists]
    db['terms.timeseries'].remove()
    db['terms.timeseries'].insert_many(docs)

def nn(self, model_type, n=20):
    """
    Create terms.neighbors.

    Args:
        model_type: {'full', 'decades', 'authors', 'locations'}
        n: the number of top nearest neighbors to store
    """
    # TODO load models  + model names from somwhere

    if model_type == 'full':
        termdict = {'full': {'neighbors': [], 'scores': []}}
    else:
        termdict = {}
        termdict[model_type] = {
            name: {'neighbors': [], 'scores': []} for name in model_names
        }

    for model in models:
        for term in model.wv.vocab.keys:
            if term not in nndict:
                nndict[term] = termdict
                nndict[term]['_id'] = term
            results = model.wv.vocab.keys.most_similar(word, topn=n)
            for result in results:
                nn = nndict[term][model_type]
                if model_type != 'full'
                    nn = nn[model_name]
                nn.neighbors.append(result[0])
                nn.scores.append(result[1])

    docs = list(nndict.values())
    update = {'$set': docs}
    db['terms.neighbors'].update_many({}, update)

def freq(self, model_type):
    """
    Create terms.frequencies.

    Args:
        model_type: {'full', 'decades', 'authors', 'locations'}
    """
    # TODO load models  + model names from somwhere

    if model_type == 'full':
        termdict = {'full': {'freq': None, 'relFreq': None}}
    else:
        termdict = {}
        termdict[model_type] = {
            name: {'freq': None, 'relFreq': None} for name in model_names
        }

    for model in models:
        total = sum(model.wv.vocab[term].count for term, vocab_obj in
                    model_wv.vocab.items())
        for term, vocab_obj in model_wv.vocab.items():
            if term not in nndict:
                freqdict[term] = termdict
                freqdict[term]['_id'] = term
            count = model.wv.vocab[term].count
            f = nndict[term][model_type]
            if model_type != 'full'
                f = nn[model_name]
            f.freq = count
            f.relFreq = count / total

    docs = list(freqdict.values())
    update = {'$set': docs}
    db['terms.frequencies'].update_many({}, update)
