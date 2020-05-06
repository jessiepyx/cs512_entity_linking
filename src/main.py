import models
from dataset import Dataset, KnowledgeBase


if __name__ == '__main__':
    model = models.RandomModel()
    trainset = Dataset.get('train')
    model.fit(trainset)
    print('【Random Model】')
    for dsname in Dataset.ds2path.keys():
        ds = Dataset.get(dsname)
        pred_cids = model.predict(ds)
        print(dsname, ds.eval(pred_cids))
    print('--------------------')

    model = models.PriorModel()
    trainset = Dataset.get('train')
    model.fit(trainset)
    print('【Prior Probability Model】')
    for dsname in Dataset.ds2path.keys():
        ds = Dataset.get(dsname)
        pred_cids = model.predict(ds)
        print(dsname, ds.eval(pred_cids))
    print('--------------------')

    model = models.SupModel()
    trainset = Dataset.get('train')
    model.fit(trainset)
    print('【Supervised Model】')
    for dsname in Dataset.ds2path.keys():
        ds = Dataset.get(dsname)
        pred_cids = model.predict(ds)
        print(dsname, ds.eval(pred_cids))
    print('--------------------')

    model = models.SupModelWithEmbedding()
    trainset = Dataset.get('train')
    model.fit(trainset)
    print('【Supervised Model With Entity Embeddings】')
    for dsname in Dataset.ds2path.keys():
        ds = Dataset.get(dsname)
        pred_cids = model.predict(ds)
        print(dsname, ds.eval(pred_cids))
    print('--------------------')

    model = models.SupModelNN()
    trainset = Dataset.get('train')
    model.fit(trainset)
    print('【Supervised Model Using Nearual Nets】')
    for dsname in Dataset.ds2path.keys():
        ds = Dataset.get(dsname)
        pred_cids = model.predict(ds)
        print(dsname, ds.eval(pred_cids))
    print('--------------------')

    model = models.MyModel()
    trainset = Dataset.get('train')
    knowlege_base = KnowledgeBase.get('train')
    model.fit(trainset, knowlege_base)
    print('【My Model】')
    for dsname in Dataset.ds2path.keys():
        ds = Dataset.get(dsname)
        kb = KnowledgeBase.get(dsname)
        pred_cids = model.predict(ds, kb)
        print(dsname, ds.eval(pred_cids))
    print('--------------------')

    model = models.MyModelNN()
    trainset = Dataset.get('train')
    knowlege_base = KnowledgeBase.get('train')
    model.fit(trainset, knowlege_base)
    print('【My Model With NN】')
    for dsname in Dataset.ds2path.keys():
        ds = Dataset.get(dsname)
        kb = KnowledgeBase.get(dsname)
        pred_cids = model.predict(ds, kb)
        print(dsname, ds.eval(pred_cids))
    print('--------------------')
