from collections import namedtuple
import torch
import gc

from .factory import *


BenchResult = namedtuple('BenchResult', [
    'name', 'avg_fwd', 'std_fwd', 'avg_bwd', 'std_bwd',
])


def fit_str(colwidth=8):
    if len(str) < colwidth:
        return (colwidth - len(str)) * ' ' + str
    else:
        return str[:colwidth]


def to_str(item):
    if isinstance(item, float):
        return '%.4g' % item
    return str(item)


def print_header(colwidth=8):
    items = []
    for item in BenchResult._fields():
        items.append(fit_str(item))
    return ''.join(items)


def pretty_print(benchresult, colwidth=8):
    items = []
    for thing in BenchResult:
        items.append(fit_str(to_str(thing)))
    return ''.join(items)


def trainbench(name, rnn_creator, nloops=100,
               seqLength=100, numLayers=1, inputSize=512, hiddenSize=512,
               miniBatch=64, device='cuda', seed=None):
    def train_batch(rnn, inputs, params):
        # CUDA events for timing
        fwd_start_event = torch.cuda.Event(enable_timing=True)
        fwd_end_event = torch.cuda.Event(enable_timing=True)
        bwd_start_event = torch.cuda.Event(enable_timing=True)
        bwd_end_event = torch.cuda.Event(enable_timing=True)

        gc.collect()

        fwd_start_event.record()
        output, hiddens = rnn(inputs)
        fwd_end_event.record()

        grads = torch.rand_like(output)
        gc.collect()

        bwd_start_event.record()
        output.backward(grads)
        bwd_end_event.record()

        for param in params:
            param.grad.data.zero_()

        torch.cuda.synchronize()

        fwd_time = fwd_start_event.elapsed_time(fwd_end_event)
        bwd_time = bwd_start_event.elapsed_time(bwd_end_event)
        return fwd_time, bwd_time

    assert device == 'cuda'
    creator_args = dict(seqLength=seqLength, numLayers=numLayers,
                        inputSize=inputSize, hiddenSize=hiddenSize,
                        miniBatch=miniBatch, device=device, seed=seed)
    rnn, inputs, params = rnn_creator(**creator_args)

    results = [train_batch(rnn, inputs, params) for _ in range(nloops)]
    fwd_times, bwd_times = zip(*results)

    fwd_times = torch.tensor(fwd_times)
    bwd_times = torch.tensor(bwd_times)

    return BenchResult(name=name,
                       avg_fwd=fwd_times.mean(), std_fwd=fwd_times.std(),
                       avg_bwd=bwd_times.mean(), std_bwd=bwd_times.std())


class DisableCuDNN():
    def __enter__(self):
        self.saved = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

    def __exit__(self, *args, **kwargs):
        torch.backends.cudnn.enabled = self.saved


class DummyContext():
    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    params = dict(nloops=100,
                  seqLength=100, numLayers=1, inputSize=512, hiddenSize=512,
                  miniBatch=64, device='cuda', seed=None)

    to_bench = [
        ('cudnn', pytorch_lstm_creator, None),
        ('aten', pytorch_lstm_creator, DisableCuDNN),
        ('jit_flat', script_lstm_flat_inputs_creator, None),
        ('jit', script_lstm_creator, None),
    ]

    print_header()
    for name, creator, context in to_bench:
        if context is None:
            context = DummyContext
        with context():
            result = trainbench(name, creator, **params)
            pretty_print(result)
