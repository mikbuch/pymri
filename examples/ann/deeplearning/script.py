with open('conv.yaml', 'r') as f:
    train = f.read()

train_params = {'train_stop': 50000,
    'valid_stop': 871,
    'test_stop': 131,
    'batch_size': 10,
    'output_channels_h2': 64, 
    'output_channels_h3': 64,  
    'max_epochs': 500,
    'save_path': '.'}
train = train % (train_params)

from pylearn2.config import yaml_parse
train = yaml_parse.load(train)
train.main_loop()
