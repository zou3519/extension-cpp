from collections import namedtuple
import torch
import gc

from .factory import pytorch_lstm_creator, script_lstm_creator
from .factory import script_lstm_flat_inputs_creator


BenchResult = namedtuple('BenchResult', [
    'name', 'avg_fwd', 'std_fwd', 'avg_bwd', 'std_bwd',
])


RNNRunner = namedtuple('RNNRunner', [
    'name', 'creator', 'context',
])


def fit_str(string, colwidth=8):
    if len(string) < colwidth:
        return (colwidth - len(string)) * ' ' + string
    else:
        return string[:colwidth]


def to_str(item):
    if isinstance(item, float):
        return '%.4g' % item
    return str(item)


def print_header(colwidth=8, sep=' '):
    items = []
    for item in BenchResult._fields:
        items.append(fit_str(item))
    return sep.join(items)


def pretty_print(benchresult, colwidth=8, sep=' '):
    items = []
    for thing in benchresult:
        items.append(fit_str(to_str(thing)))
    return sep.join(items)


def trainbench(name, rnn_creator, nloops=100, warmup=10,
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
        output, hiddens = rnn(*inputs)
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

    [train_batch(rnn, inputs, params) for _ in range(warmup)]

    results = [train_batch(rnn, inputs, params) for _ in range(nloops)]
    fwd_times, bwd_times = zip(*results)

    fwd_times = torch.tensor(fwd_times)
    bwd_times = torch.tensor(bwd_times)

    return BenchResult(name=name,
                       avg_fwd=fwd_times.mean().item(),
                       std_fwd=fwd_times.std().item(),
                       avg_bwd=bwd_times.mean().item(),
                       std_bwd=bwd_times.std().item())


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


class AssertNoJIT():
    def __enter__(self):
        import os
        enabled = os.environ.get('PYTORCH_JIT', 1)
        assert not enabled

    def __exit__(self, *args, **kwargs):
        pass


def bench(rnn_runners, sep=' ', **params):
    print(print_header(sep=sep))
    for name, creator, context in rnn_runners:
        if context is None:
            context = DummyContext
        with context():
            result = trainbench(name, creator, **params)
            print(pretty_print(result, sep=sep))


def bench_single_layer(rnn_runners, sep=' '):
    print('Benchmarking single layer lstm...')
    params = dict(nloops=200,
                  seqLength=100, numLayers=1, inputSize=512, hiddenSize=512,
                  miniBatch=64, device='cuda', seed=None)
    bench(rnn_runners, sep=sep, **params)
    print('')


def bench_multi_layer(rnn_runners, sep=' '):
    print('Benchmarking multi layer lstm...')
    params = dict(nloops=200,
                  seqLength=100, numLayers=4, inputSize=512, hiddenSize=512,
                  miniBatch=64, device='cuda', seed=None)
    bench(rnn_runners, sep=sep, **params)
    print('')


def get_rnn_runners(*names):
    return [rnn_runners[name] for name in names]


rnn_runners = {
    'cudnn': RNNRunner('cudnn', pytorch_lstm_creator, None),
    'aten': RNNRunner('aten', pytorch_lstm_creator, DisableCuDNN),
    'jit_flat': RNNRunner('jit_flat', script_lstm_flat_inputs_creator, None),
    'jit': RNNRunner('jit', script_lstm_creator, None),
    'py': RNNRunner('py', script_lstm_creator, AssertNoJIT),
    'pyflat': RNNRunner('pyflat', script_lstm_flat_inputs_creator, AssertNoJIT),
}


if __name__ == '__main__':
    rnn_runners = get_rnn_runners('cudnn', 'aten', 'jit_flat', 'jit')

    bench_single_layer(rnn_runners)
    bench_multi_layer(rnn_runners)
