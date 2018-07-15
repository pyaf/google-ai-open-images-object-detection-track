params = {}

# general
params['num_classes'] = 600
params['num_anchors'] = 9
params['input_size'] = 256
params['batch_size'] = 8
# encoder.py

params['anchor_areas'] = [8*8., 16*16., 32*32., 64*64., 128*128]
params['aspect_ratios'] = [1/2., 1/1., 2/1.]
params['scale_ratios'] = [1., pow(2, 1/3.), pow(2, 2/3.)]

params['IOU_THRESH'] = 0.2
params['CLS_THRESH'] = 0.3
params['NMS_THRESH'] = 0.3


# loss.py
params['alpha'] = 0.25
params['gamma'] = 2