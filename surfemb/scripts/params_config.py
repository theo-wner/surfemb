# Initialize argument parser
import argparse
parser = argparse.ArgumentParser(description=None)
parser.add_argument('--grayscale', action='store_true')

parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--device', type=str, default='cuda:1', help='Device')
parser.add_argument('--num-workers', type=int, default=32, help='Number of workers')

parser.add_argument('--res-crop', type=int, default=224)
parser.add_argument('--max-steps', type=int, default=500_000)

# Parse arguments
args = parser.parse_args()

# Set the arguments
GRAYSCALE = args.grayscale
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
DEVICE = args.device
DEVICE_NUM = int(DEVICE.split(':')[1])
NUM_WORKERS = args.num_workers
RES_CROP = args.res_crop
MAX_STEPS = args.max_steps