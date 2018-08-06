import time
import gc
import torch
import torch.nn.functional as F


@torch.jit.script
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


def milstm_jit(x, hx, cx, w_ih, w_hh, alpha, beta_i, beta_h, bias):
    output = []
    hy = hx[0]
    cy = cx[0]
    for j in range(x.size(0)):
        # assume 0 + 1 layers
        hy, cy = milstm_raw(x[j], hy, cy, w_ih, w_hh,
                            alpha, beta_i, beta_h, bias)
        output.append(hy)
    return torch.stack(output), (hy, cy)


def milstm_input(seqLength=100, numLayers=1, hiddenSize=512, miniBatch=64):
    x = torch.randn(seqLength, miniBatch, hiddenSize, device='cuda')
    hx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')
    cx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')

    lstm = torch.nn.LSTM(hiddenSize, hiddenSize, numLayers).cuda()
    extras = [torch.randn(numLayers * 4 * hiddenSize, device='cuda', requires_grad=True),
              torch.randn(numLayers * 4 * hiddenSize, device='cuda', requires_grad=True),
              torch.randn(numLayers * 4 * hiddenSize, device='cuda', requires_grad=True)]
    wih, whh, bih, bhh = lstm.all_weights[0]
    return x, hx, cx, wih, whh, extras[0], extras[1], extras[2], bih


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
        if (r - e).norm() > 0.0001:
            barf()


def new_params(*params):
    return [p.clone().detach().requires_grad_(True) for p in params]


def test(seqLength=100, numLayers=1, hiddenSize=512, miniBatch=64,
         check_grad=True):
    x = torch.randn(seqLength, miniBatch, hiddenSize, device='cuda')
    hx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')
    cx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')

    lstm = torch.nn.LSTM(hiddenSize, hiddenSize, numLayers).cuda()
    params = lstm.all_weights[0]
    expected = lstm(x, (hx, cx))

    jit_params = new_params(*lstm.all_weights[0])
    result = lstm_jit(x, (hx, cx), *jit_params)
    check_output(result, expected)

    grad = torch.randn_like(expected[0])
    expected[0].backward(grad)
    result[0].backward(grad)

    for r, e in zip(jit_params, params):
        if (r.grad - e.grad).max() > 0.01:
            barf()


def lstm_input(seqLength=100, numLayers=1, hiddenSize=512, miniBatch=64):
    x = torch.randn(seqLength, miniBatch, hiddenSize, device='cuda')
    hx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')
    cx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')
    lstm = torch.nn.LSTM(hiddenSize, hiddenSize, numLayers).cuda()
    wih, whh, bih, bhh = lstm.all_weights[0]
    return x, (hx, cx), wih, whh, bih, bhh


def lstm_module_input(seqLength=100, numLayers=1, hiddenSize=512, miniBatch=64):
    x = torch.randn(seqLength, miniBatch, hiddenSize, device='cuda')
    hx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')
    cx = torch.randn(numLayers, miniBatch, hiddenSize, device='cuda')
    lstm = torch.nn.LSTM(hiddenSize, hiddenSize, numLayers).cuda()
    return lstm, x, (hx, cx)


def lstm_thnn(lstm, x, hiddens):
    torch.backends.cudnn.enabled = False
    result = lstm(x, hiddens)
    torch.backends.cudnn.enabled = True
    return result


def lstm_cudnn(lstm, x, hiddens):
    return lstm(x, hiddens)


def benchmark(seqLength=100, numLayers=1, hiddenSize=512, miniBatch=64,
              nloops=100, warmup=10, sleep_between=0):
    def bench_factory(fn, input_lambd):
        def helper():
            # CUDA events for timing
            fwd_start_event = torch.cuda.Event(enable_timing=True)
            fwd_end_event = torch.cuda.Event(enable_timing=True)
            bwd_start_event = torch.cuda.Event(enable_timing=True)
            bwd_end_event = torch.cuda.Event(enable_timing=True)

            inputs = input_lambd(seqLength, numLayers, hiddenSize, miniBatch)
            gc.collect()
            torch.cuda.synchronize()

            # Time forward pass
            fwd_start_event.record()
            y, (hy, cy) = fn(*inputs)
            fwd_end_event.record()

            gc.collect()
            grads = torch.rand_like(y)

            bwd_start_event.record()
            y.backward(grads)
            bwd_end_event.record()
            torch.cuda.synchronize()

            fwd_time = fwd_start_event.elapsed_time(fwd_end_event)
            bwd_time = bwd_start_event.elapsed_time(bwd_end_event)
            return fwd_time, bwd_time

        return helper

    def benchmark(fn, input_lambd):
        time.sleep(sleep_between)
        fwd_timings = []
        bwd_timings = []
        bench = bench_factory(fn, input_lambd)
        for i in range(warmup):
            bench()

        for i in range(nloops):
            fwd_msecs, bwd_msecs = bench()
            fwd_timings.append(fwd_msecs)
            bwd_timings.append(bwd_msecs)
        return fwd_timings, bwd_timings

    def summarise(timings):
        return '%.4g' % (sum(timings) / len(timings))

    def disp_timing(name, timings):
        fwd_timings, bwd_timings = timings
        return (name,
                summarise(fwd_timings),
                summarise(bwd_timings))

    outs = [
        ("thnn", benchmark(lstm_thnn, lstm_module_input)),
        ("cudnn", benchmark(lstm_cudnn, lstm_module_input)),
        ("jit", benchmark(lstm_jit, lstm_input)),
        ("mi-j", benchmark(milstm_jit, milstm_input)),
    ]

    cols = [("algo", "fwd", "bwd")]
    cols.extend([disp_timing(*out) for out in outs])

    rows = [*zip(*cols)]
    for row in rows:
        print('\t'.join(row))


# Initialize cuda things...
x = torch.randn(3, 3, device='cuda')
y = torch.randn(3, 3, device='cuda')
z = x @ y

test()

allocated = torch.cuda.memory_allocated()

sizes = dict(seqLength=100, numLayers=1, hiddenSize=512, miniBatch=64,
             nloops=5, warmup=1, sleep_between=1)
benchmark(**sizes)

now_allocated = torch.cuda.memory_allocated()
if allocated < now_allocated:
    print("leaked: {} bytes".format(now_allocated - allocated))
    exit(1)
