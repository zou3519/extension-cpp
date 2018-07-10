import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import lltm_cuda

i = torch.randn(1).cuda()
torch.manual_seed(42)

# seqLength = 1
# numLayers = 1
# hiddenSize = 2
# miniBatch = 1
# numElements = miniBatch * hiddenSize


def lstm_kernel(x, hx, cx, w, b, perfopts=31):
    numLayers = hx.size(0)
    seqLength = x.size(0)

    # Hopefully the overhead of this is minimal
    x_ = x.unsqueeze(0).repeat(numLayers + 1, 1, 1, 1).contiguous()
    hx_ = hx.unsqueeze(1).repeat(1, seqLength + 1, 1, 1).contiguous()
    cx_ = cx.unsqueeze(1).repeat(1, seqLength + 1, 1, 1).contiguous()

    result = lltm_cuda.lstm(x_, hx_, cx_, w, b, perfopts)
    y = result[0][-1]
    hy = result[1][:, -1, :, :]
    cy = result[2][:, -1, :, :]
    return y, (hy, cy)


def flatten_weights(all_weights):
    numLayers = len(all_weights)
    whh = all_weights[0][1]
    hiddenSize = whh.size(1)
    w = torch.zeros(numLayers, 2, 4 * hiddenSize, hiddenSize, device='cuda')
    b = torch.zeros(numLayers, 2, 4 * hiddenSize, device='cuda')
    for layer in range(len(all_weights)):
        w_ih, w_hh, b_ih, b_hh = all_weights[layer]
        w[layer][0].copy_(w_ih)
        w[layer][1].copy_(w_hh)
        b[layer][0].copy_(b_ih)
        b[layer][1].copy_(b_hh)
    return w.view(-1), b.view(-1)


# run one step of an lstm, assuming premultiplied input
def lstm_cell(input_, hx, cx, w_hh, b_hh):
    gates = input_ + hx.mm(w_hh.t()) + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy


script_cell = torch.jit.script(lstm_cell)

BLOCK_SIZE = 8


# run BLOCK_SIZE steps, (possibly) compiled into a trace
def lstm_block(input_, hx, cx, w_hh, b_hh, jit=True):
    output = []
    if jit:
        cell_fn = script_cell
    else:
        cell_fn = lstm_cell
    for i in range(BLOCK_SIZE):
        hx, cx = cell_fn(input_[i], hx, cx, w_hh, b_hh)
        output.append(hx)
    return output, cx


def lstm_jit(input, hidden, w_ih, w_hh, b_ih, b_hh, jit=True):
    if jit:
        cell_fn = script_cell
    else:
        cell_fn = lstm_cell
    hx, cx = hidden[0][0], hidden[1][0]
    seq_len, batch_size, input_size = input.size()
    # pre-multiply the inputs
    input_ = F.linear(input.view(-1, input_size), w_ih, b_ih).view(seq_len,
                                                                   batch_size,
                                                                   -1)
    output = []
    traced_block = lstm_block
    for i in range(0, input.size(0), BLOCK_SIZE):
        if i + BLOCK_SIZE <= input.size(0):
            if traced_block is None:
                traced_block = torch.jit.trace(input_.narrow(0, i, BLOCK_SIZE),
                                               hx, cx, w_hh, b_hh)(lstm_block)
            # execute an entire block
            o, cx = traced_block(input_.narrow(0, i, BLOCK_SIZE), hx, cx,
                                 w_hh, b_hh, jit)
            hx = o[-1]
            output += o
        else:
            # a block doesn't fit the remaining sequence, so just
            # use the unblocked version for the end
            for ii in range(i, min(i + BLOCK_SIZE, input.size(0))):
                hx, cx = cell_fn(input_[ii], hx, cx, w_hh, b_hh)
                output.append(hx)
    output = torch.cat(output, 0).view(input_.size(0), *output[0].size())

    # to see details about the trace, unncomment:
    # print(lstm_cell.jit_debug_info())

    return output, (hx.view(1, *hx.size()), cx.view(1, *cx.size()))


def lstm_raw(input, hx, cx, w_ih, w_hh, b_ih, b_hh):
    cell_fn = lstm_cell
    seq_len, batch_size, input_size = input.size()
    # pre-multiply the inputs
    input_ = F.linear(input.view(-1, input_size), w_ih, b_ih).view(seq_len,
                                                                   batch_size,
                                                                   -1)
    output = []
    for i in range(0, input.size(0)):
        hx, cx = cell_fn(input_[i], hx, cx, w_hh, b_hh)
        output.append(hx)
    output = torch.cat(output, 0).view(input_.size(0), *output[0].size())
    return output, hx.view(1, *hx.size()), cx.view(1, *cx.size())


def lstm_cell_no_premul(x, hx, cx, w_ih, w_hh, b_ih, b_hh, transpose):
    if transpose:
        gates = x.mm(w_ih.t()) + hx.mm(w_hh.t()) + b_ih + b_hh
    else:
        gates = x.mm(w_ih) + hx.mm(w_hh) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy


def lstm_raw_no_premul(input, hx, cx, w_ih, w_hh, b_ih, b_hh, transpose):
    seq_len, batch_size, input_size = input.size()
    output = []
    for i in range(0, input.size(0)):
        hx, cx = lstm_cell_no_premul(input[i], hx, cx, w_ih, w_hh, b_ih, b_hh,
                                     transpose)
        output.append(hx)
    output = torch.cat(output, 0).view(input.size(0), *output[0].size())
    return output, hx.view(1, *hx.size()), cx.view(1, *cx.size())


traced_fn = None


def lstm_trace(input, hidden, w_ih, w_hh, b_ih, b_hh):
    global traced_fn
    if traced_fn is None:
        traced_fn = torch.jit.trace(input, hidden[0][0], hidden[1][0], w_ih,
                                    w_hh, b_ih, b_hh)(lstm_raw)
    y, hy, cy = traced_fn(input, hidden[0][0], hidden[1][0],
                          w_ih, w_hh, b_ih, b_hh)
    return y, (hy, cy)


traced_fn2 = None


def lstm_trace_no_premul(input, hidden, w_ih, w_hh, b_ih, b_hh):
    transpose = torch.tensor(1)

    global traced_fn2
    if traced_fn2 is None:
        args = [input, hidden[0][0], hidden[1][0], w_ih, w_hh, b_ih, b_hh,
                transpose]
        traced_fn2 = torch.jit.trace(*args)(lstm_raw_no_premul)
    y, hy, cy = traced_fn2(input, hidden[0][0], hidden[1][0],
                           w_ih, w_hh, b_ih, b_hh, transpose)
    # args = [input, hidden[0][0], hidden[1][0], w_ih, w_hh, b_ih, b_hh]
    # graph = traced_fn2.graph_for(*args)
    # import pdb; pdb.set_trace()
    return y, (hy, cy)


traced_fn2t = None


def lstm_trace_no_premul_pret(input, hidden, w_ih, w_hh, b_ih, b_hh):
    w_ih = w_ih.t().contiguous()
    w_hh = w_hh.t().contiguous()
    transpose = torch.tensor(0)

    global traced_fn2t
    if traced_fn2t is None:
        args = [input, hidden[0][0], hidden[1][0], w_ih, w_hh, b_ih, b_hh,
                transpose]
        traced_fn2t = torch.jit.trace(*args)(lstm_raw_no_premul)
    y, hy, cy = traced_fn2t(input, hidden[0][0], hidden[1][0],
                            w_ih, w_hh, b_ih, b_hh, transpose)
    # args = [input, hidden[0][0], hidden[1][0], w_ih, w_hh, b_ih, b_hh]
    # graph = traced_fn2t.graph_for(*args)
    # import pdb; pdb.set_trace()
    return y, (hy, cy)


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
    w, b = flatten_weights(lstm.all_weights)

    expected = lstm(x, (hx, cx))
    # print("Test lstm kernel (pointwise)...")
    # # NB: 4 by itself is broken; requires 1
    # kernel_result = lstm_kernel(x, hx, cx, w, b, 1 | 4)
    # check_output(kernel_result, expected)

    # This is just not correct...
    # print("Test lstm kernel (base)...")
    # kernel_result = lstm_kernel(x, hx, cx, w, b, 0)
    # check_output(kernel_result, expected)

    # print("Test lstm kernel (all opts)...")
    # kernel_result = lstm_kernel(x, hx, cx, w, b, 31)
    # check_output(kernel_result, expected)

    # print("Test lstm jit...")
    # jit_result = lstm_jit(x, (hx, cx), *lstm.all_weights[0])
    # check_output(jit_result, expected)

    print("Test lstm trace...")
    trace_result = lstm_trace(x, (hx, cx), *lstm.all_weights[0])
    check_output(trace_result, expected)

    print("Test lstm trace (no premul, pret)...")
    trace_np_result = lstm_trace_no_premul_pret(x, (hx, cx), *lstm.all_weights[0])
    check_output(trace_np_result, expected)
    print("Test lstm trace (no premul)...")
    trace_np_result2 = lstm_trace_no_premul(x, (hx, cx), *lstm.all_weights[0])
    check_output(trace_np_result2, expected)


def benchmark(seqLength=100, numLayers=1, hiddenSize=512, miniBatch=64):
    x = torch.randn(seqLength, miniBatch, hiddenSize, device='cuda')
    hx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')
    cx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')
    # x.requires_grad_()
    # hx.requires_grad_()
    # cx.requires_grad_()

    lstm = torch.nn.LSTM(hiddenSize, hiddenSize, numLayers).cuda()
    w, b = flatten_weights(lstm.all_weights)

    def lstmk(perfopts=1 | 2 | 4 | 8 | 16):
        result = lstm_kernel(x, hx, cx, w, b, perfopts)
        return result

    def lstmc():
        result = lstm(x, (hx, cx))
        return result

    def lstmp():
        torch.backends.cudnn.enabled = False
        result = lstm(x, (hx, cx))
        torch.backends.cudnn.enabled = True
        return result

    def lstmj():
        result = lstm_jit(x, (hx, cx), *lstm.all_weights[0])
        return result

    def lstmo():
        result = lstm_jit(x, (hx, cx), *lstm.all_weights[0], jit=False)
        return result

    def lstmt():
        return lstm_trace(x, (hx, cx), *lstm.all_weights[0])

    def lstmtnp(pretranspose=False):
        if pretranspose:
            return lstm_trace_no_premul_pret(x, (hx, cx), *lstm.all_weights[0])
        return lstm_trace_no_premul(x, (hx, cx), *lstm.all_weights[0])

    def benchmark(fn, nloops=100, warmup=2):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        timings = []
        for i in range(warmup):
            fn()
        for i in range(nloops):
            start_event.record()
            fn()
            end_event.record()
            torch.cuda.synchronize()
            gpu_msecs = start_event.elapsed_time(end_event)
            timings.append(gpu_msecs)
        return "%4.4f" % (sum(timings) / len(timings))

    # print(benchmark(lambda: lstmk(1 | 4), nloops=1, warmup=0))
    # print(benchmark(lstmp, nloops=1, warmup=0))
    # print(benchmark(lstmj, nloops=1, warmup=0))
    # print(benchmark(lstmt, nloops=1, warmup=1))
    print(benchmark(lambda: lstmtnp(True), nloops=1, warmup=1))
    return

    outs = [
        # benchmark(lstmp),
        # benchmark(lstmo),
        # benchmark(lstmc),
        # benchmark(lambda: lstmk(0)),
        benchmark(lambda: lstmk(1 | 4)),
        # benchmark(lambda: lstmk(31)),
        # benchmark(lstmj),
        # benchmark(lstmt),
        benchmark(lambda: lstmtnp(False)),
        benchmark(lambda: lstmtnp(True))
    ]
    print(', '.join(outs))

    # print("lstm (autograd)")
    # print(benchmark(lstmp))
    # print("lstm (manual)")
    # print(benchmark(lstmo))
    # print("lstm (cudnn)")
    # print(benchmark(lstmc))
    # print("lstm kernel (base) (incorrect!)")
    # print(benchmark(lambda: lstmk(0)))
    # print("lstm kernel (pointwise)")
    # print(benchmark(lambda: lstmk(1 | 4)))
    # print("lstm kernel (scheduled)")
    # print(benchmark(lambda: lstmk(31)))
    # print("lstm (jit)")
    # print(benchmark(lstmj))


# for hiddenSize in range(128, 4096, 256):
#     inputs = dict(seqLength=512,
#                   numLayers=1,
#                   hiddenSize=hiddenSize,
#                   miniBatch=64)
#     # test(**inputs)
#     benchmark(**inputs)

inputs = dict(seqLength=5,
              numLayers=1,
              hiddenSize=512,
              miniBatch=64)
# test(**inputs)
benchmark(**inputs)
