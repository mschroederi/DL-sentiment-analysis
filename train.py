from data_loader import MovieSentimentDataset
from torch.utils.data import Dataset, DataLoader


dataset = MovieSentimentDataset(csv_file='data/train.csv')

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

if __name__ == '__main__':

    for i_batch, sample_batched in enumerate(dataloader):
    
        print(i_batch, len(sample_batched['review']),len(sample_batched['sentiment']))
        print(sample_batched['review'][0],sample_batched['sentiment'][0] )

        break
