import time
import gc
import sys

import torch
import torch.nn.functional as F


@torch.jit.script
def lstm_cell(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
    gates = x.mm(w_ih.t()) + hx.mm(w_hh.t()) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy


def lstm_jit(input, hiddens, w_ih, w_hh, b_ih, b_hh):
    seq_len, batch_size, input_size = input.size()
    output = []
    hy = hiddens[0][0]
    cy = hiddens[1][0]
    for i in range(0, input.size(0)):
        # inputs = [input[i], hy, cy, w_ih, w_hh, b_ih, b_hh]
        hy, cy = lstm_cell(input[i], hy, cy, w_ih, w_hh,
                           b_ih, b_hh)
        # import pdb; pdb.set_trace()
        output.append(hy)
    return torch.stack(output), (hy.unsqueeze(0), cy.unsqueeze(0))


def barf():
    import pdb
    pdb.set_trace()


def check_output(result, expected):
    r0, (r1, r2) = result
    e0, (e1, e2) = expected
    for r, e in [(r0, e0), (r1, e1), (r2, e2)]:
        if (r - e).norm() > 0.001:
            barf()


def test(seqLength=100, numLayers=1, hiddenSize=512, miniBatch=64):
    x = torch.randn(seqLength, miniBatch, hiddenSize, device='cuda')
    hx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')
    cx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')

    lstm = torch.nn.LSTM(hiddenSize, hiddenSize, numLayers).cuda()
    expected = lstm(x, (hx, cx))
    result = lstm_jit(x, (hx, cx), *lstm.all_weights[0])
    check_output(result, expected)


def benchmark(seqLength=100, numLayers=1, hiddenSize=512, miniBatch=64):
    x = torch.randn(seqLength, miniBatch, hiddenSize, device='cuda')
    hx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')
    cx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')

    def lstm_thnn(lstm, *args):
        torch.backends.cudnn.enabled = False
        result = lstm(x, (hx, cx))
        torch.backends.cudnn.enabled = True
        return result

    def lstm_cudnn(lstm, *args):
        return lstm(x, (hx, cx))

    def run_lstm_jit(lstm, hiddens, all_weights):
        return lstm_jit(x, hiddens, *all_weights)

    def wrap_fn(fn):
        def helper():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            lstm = torch.nn.LSTM(hiddenSize, hiddenSize, numLayers).cuda()
            # for weight in lstm.all_weights[0]:
            #     weight.requires_grad_(False)
            torch.cuda.synchronize()
            gc.collect()

            start_event.record()
            fn(lstm, (hx, cx), lstm.all_weights[0])
            end_event.record()

            torch.cuda.synchronize()
            return start_event.elapsed_time(end_event)

        return helper

    def benchmark(fn, nloops=1, warmup=1):
        time.sleep(1)
        timings = []
        lambd = wrap_fn(fn)
        for i in range(warmup):
            lambd()

        for i in range(nloops):
            gpu_msecs = lambd()
            timings.append(gpu_msecs)
        return timings

    outs = [
        ("thnn", benchmark(lstm_thnn)),
        ("cudnn", benchmark(lstm_cudnn)),
        ("jit", benchmark(run_lstm_jit)),
    ]
    descs, times = zip(*outs)
    print('\t'.join(descs))
    print('\t'.join(["%.4g" % (sum(t) / len(t)) for t in times]))


# test()


# Initialize cuda things...
x = torch.randn(3, 3, device='cuda')
y = torch.randn(3, 3, device='cuda')
z = x @ y

allocated = torch.cuda.memory_allocated()
benchmark()
now_allocated = torch.cuda.memory_allocated()
if allocated < now_allocated:
    print("leaked: {} bytes".format(now_allocated - allocated))
    exit(1)
