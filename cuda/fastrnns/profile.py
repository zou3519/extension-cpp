import time
import torch

from .runner import get_rnn_runners


def run_rnn(name, rnn_creator, nloops=5,
            seqLength=100, numLayers=1, inputSize=512, hiddenSize=512,
            miniBatch=64, device='cuda', seed=None):
    def run_iter(rnn, inputs, params):
        output, hiddens = rnn(inputs)
        grads = torch.rand_like(output)
        output.backward(grads)
        for param in params:
            param.grad.data.zero_()
        torch.cuda.synchronize()

    assert device == 'cuda'
    creator_args = dict(seqLength=seqLength, numLayers=numLayers,
                        inputSize=inputSize, hiddenSize=hiddenSize,
                        miniBatch=miniBatch, device=device, seed=seed)
    rnn, inputs, params = rnn_creator(**creator_args)

    [run_iter(rnn, inputs, params) for _ in range(nloops)]


def profile(rnns, sleep_between_seconds=1, nloops=5,
            seqLength=100, numLayers=1, inputSize=512, hiddenSize=512,
            miniBatch=64, device='cuda', seed=None):
    params = dict(seqLength=seqLength, numLayers=numLayers,
                  inputSize=inputSize, hiddenSize=hiddenSize,
                  miniBatch=miniBatch, device=device, seed=seed)
    for name, creator, context in get_rnn_runners(*rnns):
        with context():
            run_rnn(name, creator, nloops, **params)
            time.sleep(sleep_between_seconds)


if __name__ == '__main__':
    # TODO: shell out to nvprof, save to dropbox?
    profile('cudnn', 'aten', 'jit_flat')
