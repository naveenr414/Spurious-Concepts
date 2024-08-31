import pickle

for fold in ["train","val","test"]:
    orig = pickle.load(open("{}_old.pkl".format(fold),"rb"))
    for i in orig:
        i['class_label'] = int((i['attribute_label'][0] + i['attribute_label'][5]+i['attribute_label'][10]) > 0)

    pickle.dump(orig,open("{}.pkl".format(fold),"wb"))