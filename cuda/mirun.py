import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import timeit

import lltm_cuda

# Benchmark assumes input_size == hidden_size
i = torch.randn(1).cuda()
torch.manual_seed(42)


# NB: 4 by itself is broken, requires 1 | 4
def milstm_kernel(x, hx, cx, w, b, alpha, beta1, beta2, perfopts=1 | 4):
    numLayers = hx.size(0)
    seqLength = x.size(0)

    # Hopefully the overhead of this is minimal
    x_ = x.unsqueeze(0).repeat(numLayers + 1, 1, 1, 1).contiguous()
    hx_ = hx.unsqueeze(1).repeat(1, seqLength + 1, 1, 1).contiguous()
    cx_ = cx.unsqueeze(1).repeat(1, seqLength + 1, 1, 1).contiguous()

    result = lltm_cuda.milstm(x_, hx_, cx_, w, b, alpha, beta1, beta2,
                              perfopts)
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


def milstm_raw(x, hx, cx, w_ih, w_hh, alpha, beta_i, beta_h, bias):
    Wx = x.mm(w_ih.t())
    Uz = hx.mm(w_hh.t())

    # Section 2.1 in https://arxiv.org/pdf/1606.06630.pdf
    gates = (alpha * Wx * Uz + beta_i * Wx + beta_h * Uz + bias)

    # Same as LSTMCell after this point
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = ingate.sigmoid()
    forgetgate = forgetgate.sigmoid()
    cellgate = cellgate.tanh()
    outgate = outgate.sigmoid()

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * cy.tanh()

    return hy, cy


def milstm(x, hx, cx, w_ih, w_hh, alpha, beta_i, beta_h, bias):
    output = []
    hy = hx[0]
    cy = cx[0]
    for j in range(x.size(0)):
        # assume 0 + 1 layers
        hy, cy = milstm_raw(x[j], hy, cy, w_ih, w_hh,
                            alpha, beta_i, beta_h, bias)
        output.append(hy)
    return torch.stack(output), (hy, cy)

def milstm2(x, hx, cx, w_ih, w_hh, alpha, beta_i, beta_h, bias):
    output = []
    hy = hx
    cy = cx
    for j in range(x.size(0)):
        # assume 0 + 1 layers
        hy, cy = milstm_raw(x[j], hy, cy, w_ih, w_hh,
                            alpha, beta_i, beta_h, bias)
        output.append(hy)
    return torch.stack(output), hy, cy

traced_fn = None

def milstm_trace(input, hidden, w_ih, w_hh, b_ih, b_hh, alpha, beta_i, beta_h):
    global traced_fn
    if traced_fn is None:
        traced_fn = torch.jit.trace(input, hidden[0][0], hidden[1][0], w_ih,
                                    w_hh, alpha, beta_i,
                                    beta_h, b_ih + b_hh)(milstm2)
    y, hy, cy = traced_fn(input, hidden[0][0], hidden[1][0],
                          w_ih, w_hh, alpha, beta_i,
                          beta_h, b_ih + b_hh)
    return y, (hy, cy)


block_fn = None


def milstm_block_unit(input, hx, cx, w_ih, w_hh, b_ih, b_hh,
                      alpha, beta_i, beta_h):
    global block_fn
    # block_fn = milstm2
    if block_fn is None:
        block_fn = torch.jit.trace(input, hx, cx, w_ih,
                                   w_hh, alpha, beta_i,
                                   beta_h, b_ih + b_hh)(milstm2)
    y, hy, cy = block_fn(input, hx, cx,
                         w_ih, w_hh, alpha, beta_i,
                         beta_h, b_ih + b_hh)
    y2,hy2,cy2 = milstm2(input, hx, cx,
                         w_ih, w_hh, alpha, beta_i,
                         beta_h, b_ih + b_hh)
    if (y - y2).norm() > 0.01:
        barf()
    if (hy - hy2).norm() > 0.01:
        barf()
    if (cy - cy2).norm() > 0.01:
        barf()
    return y, hy, cy


def milstm_blocked_trace(input, hidden, w_ih, w_hh,
                         b_ih, b_hh, alpha, beta1, beta2,
                         block_size=100):
    seq_len, batch_size, input_size = input.size()
    output = []

    hy = hidden[0][0]
    cy = hidden[1][0]

    for i in range(0, input.size(0), block_size):
        if i + block_size <= input.size(0):
            o, hy, cy = milstm_block_unit(input.narrow(0, i, block_size),
                                          hy, cy, w_ih, w_hh,
                                          b_ih, b_hh, alpha,
                                          beta1, beta2)
            output.append(o)
        else:
            # a block doesn't fit the remaining sequence, so just
            # use the unblocked version for the end
            for ii in range(i, min(i + block_size, input.size(0))):
                hy, cy = milstm_cell(input[ii], hy, cy, w_hh, b_ih, b_hh,
                                     alpha, beta1, beta2)
                output.append(hy)

    # to see details about the trace, unncomment:
    # print(lstm_cell.jit_debug_info())
    return torch.cat(output, dim=0), (hy, cy)


# run one step of an lstm, assuming premultiplied input
@torch.jit.script
def milstm_cell(input_, hx, cx, w_hh, b_ih, b_hh, alpha, beta1, beta2):
    Wx = input_
    Uz = hx.mm(w_hh.t())
    gates = alpha * Wx * Uz + beta1 * Wx + beta2 * Uz + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy


BLOCK_SIZE = 32


# run BLOCK_SIZE steps, (possibly) compiled into a trace
def milstm_block(input_, hx, cx, w_hh, b_ih, b_hh, alpha, beta1, beta2):
    output = []
    for i in range(BLOCK_SIZE):
        hx, cx = milstm_cell(input_[i], hx, cx, w_hh, b_ih,
                             b_hh, alpha, beta1, beta2)
        output.append(hx)
    return torch.stack(output), cx


def milstm_jit(input, hidden, w_ih, w_hh, b_ih, b_hh, alpha, beta1, beta2,
               trace=False):
    hx, cx = hidden[0][0], hidden[1][0]
    seq_len, batch_size, input_size = input.size()
    # pre-multiply the inputs
    input_ = F.linear(input.view(-1, input_size), w_ih).view(seq_len,
                                                             batch_size, -1)
    output = []
    if trace:
        traced_block = None
    else:
        traced_block = milstm_block
    for i in range(0, input.size(0), BLOCK_SIZE):
        if i + BLOCK_SIZE <= input.size(0):
            # execute an entire block
            if traced_block is None:
                inputs = [input_.narrow(0, i, BLOCK_SIZE),
                          hx, cx, w_hh,
                          b_ih, b_hh, alpha,
                          beta1, beta2]
                traced_block = torch.jit.trace(*inputs)(milstm_block)
            o, cx = traced_block(input_.narrow(0, i, BLOCK_SIZE), hx, cx, w_hh,
                                 b_ih, b_hh, alpha, beta1, beta2)
            hx = o[-1]
            # Unstacking problems...
            output += [o[i] for i in range(o.size(0))]
        else:
            # a block doesn't fit the remaining sequence, so just
            # use the unblocked version for the end
            for ii in range(i, min(i + BLOCK_SIZE, input.size(0))):
                hx, cx = milstm_cell(input_[ii], hx, cx, w_hh, b_ih, b_hh,
                                     alpha, beta1, beta2)
                output.append(hx)
    output = torch.cat(output, 0).view(input_.size(0), *output[0].size())

    # to see details about the trace, unncomment:
    # print(lstm_cell.jit_debug_info())

    return output, (hx.view(1, *hx.size()), cx.view(1, *cx.size()))


def barf():
    import pdb
    pdb.set_trace()


def check_output(result, expected):
    r0, (r1, r2) = result
    e0, (e1, e2) = expected
    for r, e in [(r0, e0), (r1, e1), (r2, e2)]:
        r = r.detach()
        e = e.detach()
        if (r - e).norm() > 0.001:
            barf()


def test(seqLength=100, numLayers=1, hiddenSize=512, miniBatch=64):
    x = torch.randn(seqLength, miniBatch, hiddenSize, device='cuda')
    hx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')
    cx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')

    lstm = torch.nn.LSTM(hiddenSize, hiddenSize, numLayers).cuda()
    w, b = flatten_weights(lstm.all_weights)
    extras = [torch.randn(numLayers * 4 * hiddenSize, device='cuda'),
              torch.randn(numLayers * 4 * hiddenSize, device='cuda'),
              torch.randn(numLayers * 4 * hiddenSize, device='cuda')]
    wih, whh, bih, bhh = lstm.all_weights[0]

    expected = milstm(x, hx, cx, wih, whh, extras[0], extras[1], extras[2],
                      bih + bhh)

    print("Test milstm_kernel (pointwise)...")
    kernel_result = milstm_kernel(x, hx, cx, w, b, *extras)
    check_output(kernel_result, expected)

    print("Test milstm_kernel (scheduled)...")
    kernel_result = milstm_kernel(x, hx, cx, w, b, *extras, perfopts=31)
    check_output(kernel_result, expected)

    print("Test milstm_kernel (slow)... (DNE)")
    # kernel_result = milstm_kernel(x, hx, cx, w, b, *extras, perfopts=1)
    # check_output(kernel_result, expected)

    print("Test milstm (script)...")
    jit_result = milstm_jit(x, (hx, cx), *lstm.all_weights[0], *extras)
    check_output(jit_result, expected)

    # print("Test milstm (trace)...")
    # full_trace_result = milstm_trace(x, (hx, cx),
    #                                  *lstm.all_weights[0], *extras)
    # check_output(full_trace_result, expected)

    print("Test milstm (block trace)...")
    block_result = milstm_blocked_trace(x, (hx, cx),
                                        *lstm.all_weights[0],
                                        *extras)
    check_output(block_result, expected)

    print("Test milstm (script + block trace)...")
    trace_result = milstm_jit(x, (hx, cx), *lstm.all_weights[0],
                              *extras, trace=True)
    check_output(trace_result, expected)

    print("All tests passed.")


def benchmark(seqLength=100, numLayers=1, hiddenSize=512, miniBatch=64):
    x = torch.randn(seqLength, miniBatch, hiddenSize, device='cuda')
    hx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')
    cx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')

    lstm = torch.nn.LSTM(hiddenSize, hiddenSize, numLayers).cuda()
    w, b = flatten_weights(lstm.all_weights)
    extras = [torch.randn(numLayers * 4 * hiddenSize, device='cuda'),
              torch.randn(numLayers * 4 * hiddenSize, device='cuda'),
              torch.randn(numLayers * 4 * hiddenSize, device='cuda')]
    wih, whh, bih, bhh = lstm.all_weights[0]

    def milstmk(perfopts=1 | 4):
        result = milstm_kernel(x, hx, cx, w, b, extras[0], extras[1],
                               extras[2],
                               perfopts)
        return result

    def milstmj(trace=False):
        result = milstm_jit(x, (hx, cx), *lstm.all_weights[0],
                            *extras, trace=trace)
        return result

    def milstmt():
        return milstm_blocked_trace(x, (hx, cx), *lstm.all_weights[0],
                                    *extras)

    def milstmr():
        result = milstm(x, hx, cx, wih, whh, *extras, bih + bhh)
        return result

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
        print("%4.4f msec" % (sum(timings) / len(timings)))

    print("milstm (base)")
    benchmark(milstmr)
    # print("milstm (base kernel)")
    # benchmark(lambda: milstmk(0))
    print("milstm (pointwise kernel)")
    benchmark(milstmk)
    print("milstm (scheduled kernel)")
    benchmark(lambda: milstmk(31))
    print("milstm (jit)")
    benchmark(milstmj)
    print("milstm (blocked trace)")
    benchmark(milstmt)
    # print("milstm (full trace)")
    # benchmark(milstmt)
    # print("milstm (jit blocks)")
    # benchmark(lambda: milstmj(trace=True), nloops=20)


inputs = dict(seqLength=100,
              numLayers=1,
              hiddenSize=512,
              miniBatch=64)
test(**inputs)
benchmark(**inputs)
