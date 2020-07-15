import numpy as np

SMOOTHING_STEP_SIZE = 1000
LEFT_X_CUTOFF = -20
RIGHT_X_CUTOFF = 20

def ref_mish(x):
    return x * np.tanh(np.log1p(np.exp(x)))

def ref_grad(x):
    sp = np.log1p(np.exp(x))
    grad_sp = -np.expm1(-sp)
    tsp = np.tanh(sp)
    grad_tsp = (1 - tsp * tsp) * grad_sp
    return x * grad_tsp + tsp

def generate_stats(src):
    x_list = []
    y_list = []

    with open(src, "r") as f:
        for line in f.readlines():
            x, y = [float(field.strip()) for field in line.split(' ')]
            if LEFT_X_CUTOFF < x and x < RIGHT_X_CUTOFF:
                x_list.append(x)
                y_list.append(y)

    rel_error_log10 = []
    abs_diff_err = []

    for x, y in zip(x_list, y_list):
        x128 = np.float128(x)
        y128 = np.float128(y)
        ref = ref_mish(x128)

        diff = np.abs(y128 - ref)
        rerr = -np.Inf if diff == 0 else np.log10(np.abs(diff / ref))

        log_diff = 0 if diff == 0 else np.log10(diff)

        rel_error_log10.append(float(rerr))
        abs_diff_err.append(float(diff))

    # smoothing
    x_final = []
    rel_error_log10_final = []
    abs_diff_err_final = []

    for step in range(len(x_list) // SMOOTHING_STEP_SIZE):
        ibegin = step * SMOOTHING_STEP_SIZE
        iend = ibegin + SMOOTHING_STEP_SIZE
        
        avg_x = np.mean(x_list[ibegin : iend])
        max_rel_err_log10 = np.max(rel_error_log10[ibegin : iend])
        max_diff_err = np.max(abs_diff_err[ibegin : iend])

        x_final.append(avg_x)
        rel_error_log10_final.append(max_rel_err_log10)
        abs_diff_err_final.append(max_diff_err)

    return x_final, rel_error_log10_final, abs_diff_err_final

x1, re1, ad1 = generate_stats("dump_1")
x2, re2, ad2 = generate_stats("dump_2")
x3, re3, ad3 = generate_stats("dump_3")

import matplotlib.pyplot as plt

linewidth = 0.5

fig, ax = plt.subplots(1, 3)

labels = ["fp32", "fp16 (old)", "fp16 (new)"]

ax[0].plot(x1, re1, linewidth = linewidth, c = 'g', label = labels[0])
ax[0].plot(x2, re2, linewidth = linewidth, c = 'r', label = labels[1])
ax[0].plot(x3, re3, linewidth = linewidth, c = 'b', label = labels[2])
ax[0].set_title("relative error (log10)")
ax[0].legend()

ax[1].plot(x1, ad1, linewidth = linewidth, c = 'g', label = labels[0])
ax[1].plot(x2, ad2, linewidth = linewidth, c = 'r', label = labels[1])
ax[1].plot(x3, ad3, linewidth = linewidth, c = 'b', label = labels[2])
ax[1].set_title("abs(diff)")
ax[1].legend()

ax[2].plot(x1, [np.log10(a) for a in ad1], linewidth = linewidth, c = 'g', label = labels[0])
ax[2].plot(x2, [np.log10(a) for a in ad2], linewidth = linewidth, c = 'r', label = labels[1])
ax[2].plot(x3, [np.log10(a) for a in ad3], linewidth = linewidth, c = 'b', label = labels[2])
ax[2].set_title("log10(abs(diff))")
ax[2].legend()

plt.show()