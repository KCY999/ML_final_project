import matplotlib.pyplot as plt
import numpy as np

test_idxs = [i for i in range(1,6)]
acc_results = [0.57392, 0.56893, 0.57724, 0.57308, 0.58056]
f1_results = [0.59136, 0.59551, 0.58803, 0.59302, 0.59219]

print("acc_results mean: ", np.mean(acc_results))
print("f1_results mean: ", np.mean(f1_results))

plt.figure(figsize=(8, 6))
plt.plot(test_idxs, acc_results, marker='o', label='Accuracy')
plt.plot(test_idxs, f1_results, marker='s', label='F1-Macro')

plt.xlabel('Test Index')
plt.ylabel('Score')
plt.title('S2: Accuracy and F1-Macro Scores')
plt.legend()
plt.grid(True)


plt.savefig("s2_rf_acc_f1.png")
plt.show()