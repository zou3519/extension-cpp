import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import lltm_cuda

i = torch.randn(1).cuda()
torch.manual_seed(42)

seqLength = 200
numLayers = 1
hiddenSize = 512
miniBatch = 128
numElements = miniBatch * hiddenSize

def lstm_kernel(x, hx, cx, w, b):
    # Hopefully the overhead of this is minimal
    x_ = x.unsqueeze(0).repeat(numLayers + 1, 1, 1, 1).contiguous()
    hx_ = hx.unsqueeze(1).repeat(1, seqLength + 1, 1, 1).contiguous()
    cx_ = cx.unsqueeze(1).repeat(1, seqLength + 1, 1, 1).contiguous()

    result = lltm_cuda.lstm(x_, hx_, cx_, w, b)
    x_out = result[0][-1]
    hx_out = result[1][:, -1, :, :]
    cx_out = result[2][:, -1, :, :]
    y = x_out; hy = hx_out; cy = cx_out
    return y, (hy, cy)

def flatten_weights(all_weights):
    w = torch.zeros(numLayers, 2, 4 * hiddenSize, hiddenSize, device='cuda')
    b = torch.zeros(numLayers, 2, 4 * hiddenSize, device='cuda')
    for layer in range(len(all_weights)):
        w_ih, w_hh, b_ih, b_hh = all_weights[layer]
        w[layer][0].copy_(w_ih)
        w[layer][1].copy_(w_hh)
        b[layer][0].copy_(b_ih)
        b[layer][1].copy_(b_hh)
    return w.view(-1), b.view(-1)

def lstm_cudnn(x, hx, cx, w, b):
    pass


# run one step of an lstm, assuming premultiplied input
@torch.jit.script
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


BLOCK_SIZE = 8

# run BLOCK_SIZE steps, (possibly) compiled into a trace
def lstm_block(input_, hx, cx, w_hh, b_hh):
    output = []
    for i in range(BLOCK_SIZE):
        hx, cx = lstm_cell(input_[i], hx, cx, w_hh, b_hh)
        output.append(hx)
    return output, cx

def lstm_jit(input, hidden, w_ih, w_hh, b_ih, b_hh):
    hx, cx = hidden[0][0], hidden[1][0]
    seq_len, batch_size, input_size = input.size()
    # pre-multiply the inputs
    input_ = F.linear(input.view(-1, input_size), w_ih, b_ih).view(seq_len, batch_size, -1)
    output = []
    for i in range(0, input.size(0), BLOCK_SIZE):
        if i + BLOCK_SIZE <= input.size(0):
            # execute an entire block
            o, cx = lstm_block(input_.narrow(0, i, 8), hx, cx, w_hh, b_hh)
            hx = o[-1]
            output += o
        else:
            # a block doesn't fit the remaining sequence, so just
            # use the unblocked version for the end
            for ii in range(i, min(i + BLOCK_SIZE, input.size(0))):
                hx, cx = lstm_cell(input_[ii], (hx, cx), w_hh, b_hh)
                output.append(hx)
    output = torch.cat(output, 0).view(input_.size(0), *output[0].size())

    # to see details about the trace, unncomment:
    # print(lstm_cell.jit_debug_info())

    return output, (hx.view(1, *hx.size()), cx.view(1, *cx.size()))

def check_output(result, expected):
    r0, (r1, r2) = result
    e0, (e1, e2) = expected
    for r, e in [(r0, e0), (r1, e1), (r2, e2)]:
        if (r - e).norm() > 0.001:
            import pdb; pdb.set_trace()


x = torch.randn(seqLength, miniBatch, hiddenSize, device='cuda')
hx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')
cx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')

lstm = torch.nn.LSTM(hiddenSize, hiddenSize, numLayers).cuda()
w, b = flatten_weights(lstm.all_weights)
kernel_result = lstm_kernel(x, hx, cx, w, b)
jit_result = lstm_jit(x, (hx, cx), *lstm.all_weights[0])
expected = lstm(x, (hx, cx))
check_output(kernel_result, expected)
check_output(jit_result, expected)

def lstmk():
    result = lstm_kernel(x, hx, cx, w, b)
    torch.cuda.synchronize()
    return result

def lstmc():
    result = lstm(x, (hx, cx))
    torch.cuda.synchronize()
    return result

def lstmp():
    torch.backends.cudnn.enabled = False
    result = lstm(x, (hx, cx))
    torch.cuda.synchronize()
    torch.backends.cudnn.enabled = True
    return result

def lstmj():
    result = lstm_jit(x, (hx, cx), *lstm.all_weights[0])
    torch.cuda.synchronize()
    return result
