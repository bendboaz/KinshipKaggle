from implementation.DataHandling import *
from implementation.Training import PROJECT_ROOT

if __name__ == "__main__":
    datasets = ['train', 'dev', 'mini']
    data_path = os.path.join(PROJECT_ROOT, 'data')
    proc_path = os.path.join(data_path, 'processed')
    relationships_path = os.path.join(data_path, 'raw', 'train_relationships.csv')
    for dataset in datasets:
        KinshipDataset.get_dataset(os.path.join(data_path, f"{dataset}_dataset.pkl"),
                                   os.path.join(proc_path, dataset),
                                   relationships_path, force_change=True)