from torch.utils.data import DataLoader
import torch.nn as nn

from data.dataset import iclevrDataset
from prefetch_generator import BackgroundGenerator


class DataLoader_pro(DataLoader):    
    def __iter__(self):
        return BackgroundGenerator(super().__iter__());

def get_dataloader(args, device):
	"""Load i-CLEVR Dataset"""
	print("\nBuilding training & testing dataset...")

	train_dataset = iclevrDataset(args, device, "train")
	test_dataset  = iclevrDataset(args, device, "test")
	new_test_dataset  = iclevrDataset(args, device, "new_test")

	print("# training samples: {}".format(len(train_dataset)))
	print("# testing  samples: {}".format(len(test_dataset)))

	train_loader = DataLoader_pro(
		train_dataset, 
		num_workers=args.num_workers, 
		batch_size=args.batch_size, 
		shuffle=True,
		pin_memory=True
	)
	test_loader = DataLoader_pro(
		test_dataset, 
		num_workers=args.num_workers, 
		batch_size=args.batch_size, 
		shuffle=False,
		pin_memory=True
	)

	new_test_loader = DataLoader_pro(
		new_test_dataset, 
		num_workers=args.num_workers, 
		batch_size=args.batch_size, 
		shuffle=False,
		pin_memory=True
	)



	# train_loader = nn.DataParallel(train_loader)
	# test_loader = nn.DataParallel(test_loader)

	return train_loader, test_loader, new_test_loader