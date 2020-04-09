import torch
import argparse
from models import classificators


def main(model_name, input_shape):
    model = getattr(classificators, model_name)()
    for i in range(100):
        inp = torch.rand([1, 3, input_shape, input_shape])
        print(model(inp))
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=False, default='CNN', help='Choose net to check')
    parser.add_argument('--input_shape', required=False, default=512, help='Choose input shape')
    args = parser.parse_args()

    main(**vars(args))