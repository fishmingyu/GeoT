from torch_geometric import datasets
import torch
import torch.nn.functional as F
from torch_geometric.transforms import FaceToEdge
# from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np

class Dataset:
    def __init__(self, name: str, device):
        self.name = name
        self.device = device
        self.get_dataset()

    def get_dataset(self):
        if self.name == 'pubmed':
            dataset = datasets.CitationFull(root='./data/', name='PubMed')
            graph = dataset[0]
        elif self.name == 'citeseer':
            dataset = datasets.CitationFull(root='./data/', name='CiteSeer')
            graph = dataset[0]
        elif self.name == 'cora':
            dataset = datasets.CitationFull(root='./data/', name='Cora')
            graph = dataset[0]
        elif self.name == "dblp":
            dataset = datasets.CitationFull(root='./data/', name='DBLP')
            graph = dataset[0]
        elif self.name == "amazon_computers":
            dataset = datasets.Amazon(root='./data/', name='Computers')
            graph = dataset[0]
        elif self.name == "amazon_photo":
            dataset = datasets.Amazon(root='./data/', name='Photo')
            graph = dataset[0]
        elif self.name == 'ppi':
            dataset = datasets.PPI(root='./data/PPI')
            graph = dataset[0]
        elif self.name == 'reddit':
            dataset = datasets.Reddit(root='./data/Reddit')
            graph = dataset[0]
        elif self.name == 'github':
            dataset = datasets.GitHub(root='./data/GitHub')
            graph = dataset[0]
        elif self.name == 'flickr':
            dataset = datasets.Flickr(root='./data/Flickr')
            graph = dataset[0]
        elif self.name == 'yelp':
            dataset = datasets.Yelp(root='./data/Yelp')
            graph = dataset[0]
        elif self.name == 'amazon_products':
            dataset = datasets.AmazonProducts(root='./data/AmazonProducts')
            graph = dataset[0]
        elif self.name == 'fb15k_237':
            dataset = datasets.FB15k_237(root='./data/FB15k_237')
            graph = dataset[0]
        elif self.name == 'actor':
            dataset = datasets.Actor(root='./data/Actor')
            graph = dataset[0]
        elif self.name == 'airports_usa':
            dataset = datasets.Airports(root='./data/', name='USA')
            graph = dataset[0]
        elif self.name == 'airports_brazil':
            dataset = datasets.Airports(root='./data/', name='Brazil')
            graph = dataset[0]
        elif self.name == 'airports_europe':
            dataset = datasets.Airports(root='./data/', name='Europe')
            graph = dataset[0]
        elif self.name == 'malnet':
            dataset = datasets.MalNetTiny(root='./data/MalNetTiny')
            graph = dataset[0]
        elif self.name == 'emaileucore':
            dataset = datasets.EmailEUCore(root='./data/EmailEUCore')
            graph = dataset[0]
        elif self.name == 'twitch_de':
            dataset = datasets.Twitch(root='./data/', name='DE')
            graph = dataset[0]
        elif self.name == 'twitch_en':
            dataset = datasets.Twitch(root='./data/', name='EN')
            graph = dataset[0]
        elif self.name == 'twitch_es':
            dataset = datasets.Twitch(root='./data/', name='ES')
            graph = dataset[0]
        elif self.name == 'twitch_fr':
            dataset = datasets.Twitch(root='./data/', name='FR')
            graph = dataset[0]
        elif self.name == 'twitch_pt':
            dataset = datasets.Twitch(root='./data/', name='PT')
            graph = dataset[0]
        elif self.name == 'twitch_ru':
            dataset = datasets.Twitch(root='./data/', name='RU')
            graph = dataset[0]
        elif self.name == 'aifb':
            dataset = datasets.Entities(root='./data/', name='AIFB')
            graph = dataset[0]
        elif self.name == 'am':
            dataset = datasets.Entities(root='./data/', name='AM')
            graph = dataset[0]
        elif self.name == 'mutag':
            dataset = datasets.Entities(root='./data/', name='MUTAG')
            graph = dataset[0]
        elif self.name == 'bgs':
            dataset = datasets.Entities(root='./data/', name='BGS')
            graph = dataset[0]
        elif self.name == 'wikics':
            dataset = datasets.WikiCS(root='./data/WikiCS')
            graph = dataset[0]
        elif self.name == 'lastfmasia':
            dataset = datasets.LastFMAsia(root='./data/LastFMAsia')
            graph = dataset[0]
        elif self.name == 'facebookpagepage':
            dataset = datasets.FacebookPagePage(root='./data/FacebookPagePage')
            graph = dataset[0]
        elif self.name == 'cornell':
            dataset = datasets.LINKXDataset(root='./data/', name='cornell5')
            graph = dataset[0]
        elif self.name == 'genius':
            dataset = datasets.LINKXDataset(root='./data/', name='genius')
            graph = dataset[0]
        elif self.name == 'penn':
            dataset = datasets.LINKXDataset(root='./data/', name='penn94')
            graph = dataset[0]
        elif self.name == 'reed':
            dataset = datasets.LINKXDataset(root='./data/', name='reed98')
            graph = dataset[0]
        elif self.name == 'amherst':
            dataset = datasets.LINKXDataset(root='./data/', name='amherst41')
            graph = dataset[0]
        elif self.name == 'johnshopkins':
            dataset = datasets.LINKXDataset(root='./data/', name='johnshopkins55')
            graph = dataset[0]
        elif self.name == 'deezereurope':
            dataset = datasets.DeezerEurope(root='./data/DeezerEurope')
            graph = dataset[0]
        elif self.name == 'myket':
            dataset = datasets.MyketDataset(root='./data/MyketDataset')
            graph = dataset[0]
        elif self.name == 'polblogs':
            dataset = datasets.PolBlogs(root='./data/PolBlogs')
            graph = dataset[0]
        elif self.name == 'explicitbitcoin':
            dataset = datasets.EllipticBitcoinDataset(root='./data/EllipticBitcoin')
            graph = dataset[0]
        elif self.name == 'gemsecdeezer_hu':
            dataset = datasets.GemsecDeezer(root='./data/', name='HU')
            graph = dataset[0]
        elif self.name == 'gemsecdeezer_hr':
            dataset = datasets.GemsecDeezer(root='./data/', name='HR')
            graph = dataset[0]
        elif self.name == 'gemsecdeezer_ro':
            dataset = datasets.GemsecDeezer(root='./data/', name='RO')
            graph = dataset[0]
        elif self.name == 'mooc':
            dataset = datasets.JODIEDataset(root='./data/', name='MOOC')
            graph = dataset[0]
        elif self.name == 'lastfm':
            dataset = datasets.JODIEDataset(root='./data/', name='LastFM')
            graph = dataset[0]
        elif self.name == 'BrcaTcga':
            dataset = datasets.BrcaTcga(root='./data/BrcaTcga')
            graph = dataset[0]
        elif self.name == 'chameleon':
            dataset = datasets.WikipediaNetwork(root='./data/', name = 'chameleon')
            graph = dataset[0]
        elif self.name == 'squirrel':
            dataset = datasets.WikipediaNetwork(root='./data/', name = 'squirrel')
            graph = dataset[0]
        elif self.name == 'crocodile':
            dataset = datasets.WikipediaNetwork(root='./data/', name = 'crocodile', geom_gcn_preprocess=False)
            graph = dataset[0]
        else:
            raise KeyError('Unknown dataset {}.'.format(self.name))
        print("Dataset: ", self.name)
        self.edge_index = graph.edge_index.to(self.device)
        self.num_edges = graph.num_edges

        idx = self.edge_index[1]
        sorted_idx = torch.argsort(idx)
        self.idx = idx[sorted_idx]

    def idx_augment(self, idx):
        sorted_idx = torch.argsort(idx)
        return idx[sorted_idx]


if __name__ == '__main__':
    ml_datasets = ['pubmed', 'citeseer', 'cora', 'dblp', 'amazon_computers', 'amazon_photo', 'ppi', 
                   'reddit', 'github',  'fb15k_237', 'actor', 'airports_usa', 
                   'airports_brazil', 'airports_europe', 'malnet', 'emaileucore', 'twitch_de', 'twitch_en', 'twitch_es', 
                   'twitch_fr', 'twitch_pt', 'twitch_ru', 'aifb', 'am', 'mutag', 'bgs', 'wikics', 'lastfmasia', 
                   'facebookpagepage',  'cornell', 'genius', 'penn', 'reed', 'amherst', 
                   'johnshopkins', 'deezereurope', 'myket', 'polblogs', 'explicitbitcoin', 'gemsecdeezer_hu', 
                   'gemsecdeezer_hr', 'gemsecdeezer_ro', 'mooc', 'lastfm', 'BrcaTcga', 
                   'chameleon', 'squirrel', 'crocodile']


    device = "cuda"
    
    for d in ml_datasets:
        data = Dataset(d, device)
        print(d, data.num_edges, data.idx.size())
