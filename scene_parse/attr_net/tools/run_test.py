import os
import json

from options import get_options
from datasets import get_dataloader
from model import get_model
import utils


COMP_CAT_DICT_PATH = 'tools/clevr_comp_cat_dict.json'



def attribute_detection():
    attr_correct = {
    'shape': 0,
    'size': 0,
    'material': 0,
    'color': 0
    }

    attr_total = {
    'shape': 0,
    'size': 0,
    'material': 0,
    'color': 0
    }
    
    opt = get_options('test')
    test_loader = get_dataloader(opt, 'test')
    model = get_model(opt)

    if opt.use_cat_label:
        with open(COMP_CAT_DICT_PATH) as f:
            cat_dict = utils.invert_dict(json.load(f))

    if opt.dataset == 'clevr':
        scenes = [{
        'image_index': i,
        'image_filename': 'CLEVR_val_%06d.png' % i,
        'objects': []
        } for i in range(15000)]

    count = 0
    for data, labels, idxs, cat_idxs in test_loader:
        model.set_input(data)
        model.forward()
        pred = model.get_pred()
       
        for i in range(pred.shape[0]):
            if opt.dataset == 'clevr':
                img_id = idxs[i]
                obj = utils.get_attrs_clevr(pred[i])
                obj_gnd = utils.get_attrs_clevr(labels[i])
                for key in ['shape', 'size', 'material', 'color']:
                    if obj[key] == obj_gnd[key]:
                        attr_correct[key] += 1
                    attr_total[key] += 1
                
                    
                if opt.use_cat_label:
                    cid = cat_idxs[i] if isinstance(cat_idxs[i], int) else cat_idxs[i].item()
                    obj['color'], obj['material'], obj['shape'] = cat_dict[cid].split(' ')
            scenes[img_id]['objects'].append(obj)
        count += idxs.size(0)
        print('%d / %d objects processed' % (count, len(test_loader.dataset)))

    output = {  
        'info': '%s derendered scene' % opt.dataset,
        'scenes': scenes,
    }
    print('| saving annotation file to %s' % opt.output_path)
    utils.mkdirs(os.path.dirname(opt.output_path))
    for key in attr_correct:
        acc = attr_correct[key] / attr_total[key] if attr_total[key] > 0 else 0
        print(f'{key} accuracy: {acc:.4f}')
    total_correct = sum(attr_correct.values())
    total_attrs = sum(attr_total.values())

    print(f'Overall attribute accuracy: {total_correct / total_attrs:.4f}')
    with open(opt.output_path, 'w') as fout:
        json.dump(output, fout)
    return opt.output_path