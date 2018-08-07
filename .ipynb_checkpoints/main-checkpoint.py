import mxnet as mx
import argparse
from trainer import Train

def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoches', type=int, default=50)
    parser.add_argument('--GPU_COUNT', type=int, default=2)
    parser.add_argument('--show_status', type=bool, default=True)
    config = parser.parse_args()
    
    trainer = Train(config)
    
    trainer.train()

if __name__ =="__main__":
    main()