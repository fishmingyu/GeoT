from torch_geometric import datasets
from torch_geometric.utils import from_dgl
import torch
import torch.nn.functional as F
from torch_geometric.transforms import FaceToEdge
import torch_geometric.transforms as T
# from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
import os
import dgl


class Dataset:
    def __init__(self, name: str, device):
        self.name = name
        self.device = device
        self.max_exponent = 12
        self.noise_aug_num = 5
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
        elif self.name == 'flickr': # due to PyG's broken link, we use dgl's dataset
            dataset = dgl.data.FlickrDataset()
            dgl_g = dataset[0]
            graph = from_dgl(dgl_g)
        elif self.name == 'yelp':  # due to PyG's broken link, we use dgl's dataset
            dataset = dgl.data.YelpDataset()
            dgl_g = dataset[0]
            graph = from_dgl(dgl_g)
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
        elif self.name == 'faust':
            pre_transform = T.Compose([T.FaceToEdge(), T.Constant(value=1)])
            train_dataset = datasets.FAUST('./data/FAUST', True, T.Cartesian(), pre_transform)
            graph = train_dataset[0]
        elif self.name == 'modelnet10':
            pre_transform, transform = T.NormalizeScale(), T.SamplePoints(16384)
            train_dataset = datasets.ModelNet('./data/ModelNet10', '10', True, transform, pre_transform)
            graph = train_dataset[0]
        elif self.name == 'modelnet40':
            pre_transform, transform = T.NormalizeScale(), T.SamplePoints(16384)
            train_dataset = datasets.ModelNet('./data/ModelNet40', '40', True, transform, pre_transform)
            graph = train_dataset[0]
        else:
            raise KeyError('Unknown dataset {}.'.format(self.name))
        print("Dataset: ", self.name)
        self.edge_index = graph.edge_index.to(self.device)
        self.num_edges = graph.num_edges

        idx = self.edge_index[1]
        sorted_idx = torch.argsort(idx)
        self.idx = idx[sorted_idx]

    def idx_augment(self, idx):
        idx_size = idx.size(0)
        base_index = int(np.floor(np.log2(idx_size / 1000)))

        scale_idx_list = []
        for scale in range(self.max_exponent):
            scale_idx = self.augment_sample(idx, base_index, scale)
            scale_idx_list.append(scale_idx)

        return scale_idx_list


    def noise_augment(self, idx):
        torch.random.manual_seed(0)
        max_idx = idx.max()
        size = idx.size(0)
        range = max((size - max_idx) / max_idx, 1)
        device = idx.device
        noise = torch.randint(0, int(range), (size,)).to(device)
        id_noise = (idx + noise).type(torch.long)
        sorted_idx = torch.argsort(id_noise)
        return id_noise[sorted_idx]


    def augment_sample(self, idx, base_index, target_index):
        # down sample
        if target_index < base_index:
            return self.down_sample(idx, int(np.power(2, base_index - target_index)))
        elif target_index == base_index:
            return idx
        # up sample
        else:
            return self.up_sample(idx, int(np.power(2, target_index - base_index)))

    def down_sample(self, input, scale):
        input_3d = input.unsqueeze(0).unsqueeze(0)
        input_3d_float = input_3d.type(torch.float32)
        new_length = round(input.size(0) / scale)
        down_sampled = F.interpolate(input_3d_float, size=new_length, mode='nearest')
        down_sampled_1d = down_sampled.squeeze() / scale
        return torch.round(down_sampled_1d.type(torch.long))
    
    def up_sample(self, input, scale):
        input_3d = input.unsqueeze(0).unsqueeze(0)
        input_3d_float = input_3d.type(torch.float32)
        new_length = round(input.size(0) * scale)
        interpolated = F.interpolate(input_3d_float, size=new_length, mode='linear')
        interpolated_1d = interpolated.squeeze() * scale
        return torch.round(interpolated_1d.type(torch.long))
    
    def store_aug_idx(self):
        # check if the directory exists
        if not os.path.exists('./idx_data'):
            os.makedirs('./idx_data')
        for n in range(self.noise_aug_num):
            noise_idx = self.noise_augment(self.idx)
            scale_idx_list = self.idx_augment(noise_idx)
            # store the idx via torch
            for i, scale_idx in enumerate(scale_idx_list):
                print(f"Scale {i} size: {scale_idx.size()}")
                torch.save(scale_idx, f'./idx_data/{self.name}_noise{n}_{i}.pt')

if __name__ == '__main__':
    ml_datasets = ['pubmed', 'citeseer', 'cora', 'dblp', 'amazon_computers', 'amazon_photo', 'ppi', 'reddit', 'github', 
                   'flickr', 'yelp', 'fb15k_237', 'actor', 'airports_usa', 'airports_brazil', 'airports_europe',
                    'malnet', 'emaileucore', 'twitch_de', 'twitch_en', 'twitch_es', 'twitch_fr', 'twitch_pt', 'twitch_ru', 'aifb',
                    'am', 'mutag', 'bgs', 'wikics', 'lastfmasia', 'facebookpagepage', 'cornell', 'genius', 'penn', 'reed',
                    'amherst', 'johnshopkins', 'deezereurope', 'myket', 'polblogs', 'explicitbitcoin', 'gemsecdeezer_hu', 'gemsecdeezer_hr',
                    'gemsecdeezer_ro', 'mooc', 'lastfm', 'BrcaTcga', 'chameleon', 'squirrel', 'crocodile', 'faust']

    device = "cuda"
    
    for d in ml_datasets:
        data = Dataset(d, device)
        data.store_aug_idx()
