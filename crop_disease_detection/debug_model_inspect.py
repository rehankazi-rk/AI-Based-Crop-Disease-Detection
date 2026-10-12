import os
import torch

path = r'D:\B.tech\SY\SEM-III\CEP\crop_disease_detection\model\crop_disease_model.pth'
print('exists=', os.path.exists(path))
print('size=', os.path.getsize(path) if os.path.exists(path) else 'missing')
try:
    x = torch.load(path, map_location='cpu')
    print('type=', type(x))
    if hasattr(x, 'keys'):
        keys = list(x.keys())
        print('keys_count=', len(keys))
        print('keys_sample=', keys[:20])
        for k, v in x.items():
            print('first_key=', k)
            print('first_value_type=', type(v))
            print('first_shift_shape=', getattr(v, 'shape', None))
            break
except Exception as e:
    print('ERROR=', type(e).__name__, e)
