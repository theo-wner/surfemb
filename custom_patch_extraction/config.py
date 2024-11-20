import os

batch_size = 16
learning_rate = 0.001
device = 'cuda:2'
device_num = int(device.split(':')[-1])
num_workers = 32 # Max is 32