import matplotlib.pyplot as plt
import numpy as np

test_idxs = [i for i in range(1,6)]
acc_results = [0.58665, 0.58586, 0.58475, 0.59038, 0.59264]
f1_results = [0.59264, 0.59780, 0.59554, 0.59845, 0.60006]


print("acc_results mean: ", np.mean(acc_results))
print("f1_results mean: ", np.mean(f1_results))

plt.figure(figsize=(8, 6))
plt.plot(test_idxs, acc_results, marker='o', label='Accuracy')
plt.plot(test_idxs, f1_results, marker='s', label='F1-Macro')

plt.xlabel('Test Index')
plt.ylabel('Score')
plt.title('S1: Accuracy and F1-Macro Scores')
plt.legend()
plt.grid(True)


plt.savefig("s1_rf_acc_f1.png")
plt.show()