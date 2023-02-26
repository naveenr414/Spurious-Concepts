import pickle

def new_metadata(dataset,split):
    """Create a new metadata file based on the dataset, split
    
    Arguments:
        dataset: Which dataset we're looking at, such as CUB_small, etc.
        split: Train, test, or val split
        
    Returns: Nothing
    
    Side Effects: Creates a new pickle file for the appropriate dataset
    """
    
    if dataset not in ["CUB_small", "CUB_blur", "CUB_tag"]:
        raise Exception("{} not found".format(dataset))
    
    preprocessed_train = pickle.load(open("../cem/cem/CUB/preprocessed/{}.pkl".format(split),"rb"))
    preprocessed_train = [i for i in preprocessed_train if i['class_label'] <= 11]
    for i in preprocessed_train:
        i['img_path'] = i['img_path'].replace("CUB/","{}/".format(dataset))

        if i['class_label'] == 0 and dataset != 'CUB_small':
            i['attribute_label'].append(1)
        else:
            i['attribute_label'].append(0)
            
    pickle.dump(preprocessed_train,open("../cem/cem/{}/preprocessed/{}.pkl".format(dataset,split),"wb"))