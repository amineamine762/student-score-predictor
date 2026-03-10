import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from model import compute_model_output, compute_gradient,  cost_function, gradient_descent

data = pd.read_csv("student_scores.csv")
x = data['Hours_Study']
y = data['Score']

tmp_w = 0
tmp_b = 0

tmp_f_x = compute_model_output(x, tmp_w, tmp_b)

iter = 4000
alpha = 0.01
w_final, b_final, j_hist, p_hist = gradient_descent(x, y, tmp_f_x, tmp_w, tmp_b, cost_function, compute_gradient, alpha, iter)
f_x_final = compute_model_output(x, w_final, b_final)
plt.plot(x, f_x_final, c='b', label="Our prediction")
plt.scatter(x, y, marker='x',c='r', label='Actual Values')
plt.title("Predicting Student Scores")
plt.ylabel(' Exam Score ')
plt.xlabel('Stady Hours (for 1 day )')
plt.legend()
plt.savefig('prediction.png')

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(j_hist[:100])
ax2.plot(1000 + np.arange(len(j_hist[1000:])), j_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.savefig('cost_function.png')

print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
























