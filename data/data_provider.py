from torch.utils.data import DataLoader
from data.datasets import THU_006or018_basic, THU_006or018_few_shot, THU_006_generalization,DIRG_020_basic,DIRG_020_generalization
from .utils import AddNoiseTransform

DATASET_TASK_CLASS = {
    'THU_006_basic': THU_006or018_basic,
    'THU_018_basic': THU_006or018_basic,
    'THU_018_few_shot': THU_006or018_few_shot,
    'THU_006_few_shot': THU_006or018_few_shot,
    'THU_006_generalization': THU_006_generalization,
    'DIRG_020_basic': DIRG_020_basic,
    'DIRG_020_geberalization': DIRG_020_generalization
    
}

def get_data(args):
    dataset_class = DATASET_TASK_CLASS[args.dataset_task]
    
    dataset = dataset_class(args,flag = 'train')
    train_loader = DataLoader(
        dataset = dataset,
        batch_size= args.batch_size,
        shuffle = True,
        num_workers = args.num_workers
    )
    dataset = dataset_class(args,flag = 'val')
    val_loader = DataLoader(
        dataset = dataset,
        batch_size= args.batch_size,
        shuffle = False,
        num_workers = args.num_workers
    )
    dataset = dataset_class(args,flag = 'test')
    test_loader = DataLoader(
        dataset = dataset,
        batch_size= args.batch_size,
        shuffle = False,
        num_workers = args.num_workers
    )     
    return train_loader,val_loader,test_loader