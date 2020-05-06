import models
from dataset import Dataset


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
