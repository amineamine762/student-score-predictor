import numpy as np 
import copy, math

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_x = np.zeros(m)
    for i in range(m):
        f_x[i] = w * x[i] + b
    return f_x

def cost_function(f_x, y, x):
    m = x.shape[0]
    j_wb = 0
    for i in range(m):
        j_wb += (f_x[i]-y[i])**2
    j_wb = j_wb/(2*m)
    return j_wb

def compute_gradient(f_x, y, x):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        dj_dw += (f_x[i]-y[i])*x[i]
        dj_db += (f_x[i]-y[i])
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_dw,dj_db 

def gradient_descent(x, y, f_x,  w_in, b_in, cost_function, compute_gradient, alpha, num_iters):
    j_hist = []
    p_hist = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        f_x = compute_model_output(x, w, b)
        dj_dw,dj_db = compute_gradient(f_x, y, x)
        w = w - alpha*dj_dw
        b = b -alpha*dj_db

        if num_iters < 10000:
            j_hist.append(cost_function(f_x, y, x))
            p_hist.append([w,b])
        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {j_hist[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, j_hist, p_hist











    



    













