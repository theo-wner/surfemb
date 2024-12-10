import os

# Initialize argument parser
import argparse
parser = argparse.ArgumentParser(description='Custom Patch Extraction')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--device', type=str, default='cuda:2', help='Device')
parser.add_argument('--num_workers', type=int, default=32, help='Number of workers')
parser.add_argument('--grayscale', action='store_true')

# Parse arguments
args = parser.parse_args()

# Set the arguments
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
DEVICE = args.device
DEVICE_NUM = int(DEVICE.split(':')[1])
NUM_WORKERS = args.num_workers
GRAYSCALE = args.grayscale