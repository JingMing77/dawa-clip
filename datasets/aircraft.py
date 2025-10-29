import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader


template = ['a photo of a {}, a type of aircraft.']


class FGVCAircraft(DatasetBase):

    dataset_dir = 'Aircraft'

    def __init__(self, root, num_ways, num_shots, datasplit='split_aircraft.json'):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')

        self.template = template

        classnames = []
        with open(os.path.join(self.dataset_dir, 'variants.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}

        train = self.read_data(cname2lab, 'images_variant_train.txt')
        val = self.read_data(cname2lab, 'images_variant_val.txt')
        test = self.read_data(cname2lab, 'images_variant_test.txt')
        
        train = self.generate_fewshot_dataset(train, num_ways=num_ways, num_shots=num_shots)

        train, val, test = self.gen_nways_dataset(train, val, test, num_ways=num_ways)
        
        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                imname = line[0] + '.png'
                classname = ' '.join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=classname
                )
                items.append(item)
        
        return items
    
    
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