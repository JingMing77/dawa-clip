import os
import json
import random

from .utils import Datum, DatasetBase


template = ['This is a satellite photo of {}.']


class FGSCR42(DatasetBase):

    dataset_dir = 'FGSCR'

    def __init__(self, root, num_ways, num_shots, datasplit='split_fgscr42.json'):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = self.dataset_dir
        self.split_path = os.path.join(self.dataset_dir, datasplit)

        self.template = template

        train, val, test = self.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_ways=num_ways, num_shots=num_shots)
        train, val, test = self.gen_nways_dataset(train, val, test, num_ways=num_ways)
        

        super().__init__(train_x=train, val=val, test=test)

    def gen_nways_dataset(self, train, val, test, num_ways=-1):
        """
        Filter the dataset to only include num_ways classes.
        If num_ways is -1, no filtering is done.
        else, it filters the train, val, and test sets to only include the first num_ways classes.
        """
        if type(num_ways) is not int:
            num_ways = len(num_ways)
            
        if num_ways < 0:
            return train, val, test
        
        class_names = sorted(list(set([d.label for d in train])))
        label2idx = {label: idx for idx, label in enumerate(class_names)}
        num_classes = len(class_names)
        assert num_classes == num_ways, f"Number of classes in train ({num_classes}) does not match num_ways ({num_ways})"

        # Filter val and test to only include classes in train
        val = [d for d in val if d.label in class_names]
        test = [d for d in test if d.label in class_names]

        # set labels
        processed = []
        for d in train:
            if d.impath in processed:
                continue
            processed.append(d.impath)
            d.label = label2idx[d.label]
        processed = []
        for d in val:
            if d.impath in processed:
                continue
            processed.append(d.impath)
            d.label = label2idx[d.label]
        processed = []
        for d in test:
            if d.impath in processed:
                continue
            processed.append(d.impath)
            d.label = label2idx[d.label]
        return train, val, test

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                classname = classname.split('.')[-1]
                impath = os.path.join(path_prefix, impath)
                item = Datum(
                    impath=impath,
                    label=int(label),
                    classname=classname
                )
                out.append(item)
            return out
        
        print(f'Reading split from {filepath}')
        with open(filepath, 'r') as f:
            split = json.load(f)
        train = _convert(split['train'])
        val = _convert(split['val'])
        test = _convert(split['test'])

        return train, val, test
    

def process_dataset(data_dir) -> None:
    seed = 1
    random.seed(seed)  # 1,2,3,4,5,6,7,8,9,10
    # 存储数据
    train_data = []
    val_data = []
    test_data = []

    # 遍历每个类别文件夹
    for class_folder in os.listdir(data_dir):
        print(f"Processing class folder: {class_folder}")
        
        class_path = os.path.join(data_dir, class_folder)
        if os.path.isdir(class_path):
            label = int(class_folder.split('.')[0]) - 1
            images = [os.path.join(class_folder, img) for img in os.listdir(class_path) if img.endswith('.bmp')]
            # 随机打乱图片顺序
            random.shuffle(images)
            # 计算划分索引
            total = len(images)
            train_end = int(total * 0.6)
            val_end = int(total * 0.8)
            # 划分数据
            train_data.extend([[img, label, class_folder] for img in images[:train_end]])
            val_data.extend([[img, label, class_folder] for img in images[train_end:val_end]])
            test_data.extend([[img, label, class_folder] for img in images[val_end:]])
        else:
            print(f"Skipping {class_path}, not a directory.")

    # 统计每个划分的数据量
    print(f"Total train samples: {len(train_data)}")
    print(f"Total validation samples: {len(val_data)}")
    print(f"Total test samples: {len(test_data)}")
    
    # 构建JSON数据
    json_data = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

    # 保存为JSON文件
    save_path = os.path.join(data_dir, 'split_fgscr42_{}.json'.format(seed))
    with open(save_path, 'w') as f:
        json.dump(json_data, f, indent=4)

if __name__ == "__main__":
    # 设置输入和输出目录
    input_directory = r"F:\vlm\data\FGSCR"
    
    # 处理数据集
    process_dataset(input_directory)