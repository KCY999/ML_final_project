import matplotlib.pyplot as plt


test_idxs = [i for i in range(1,6)]
acc_result = [0.58259, 0.58259, 0.58259, 0.58259, 0.58259]
f1_results = [0.57391, 0.57391, 0.57391, 0.57391, 0.57391]


plt.figure(figsize=(8, 6))
plt.plot(test_idxs, acc_result, marker='o', label='Accuracy')
plt.plot(test_idxs, f1_results, marker='s', label='F1-Macro')

plt.xlabel('Test Index')
plt.ylabel('Score')
plt.title('S1: Accuracy and F1-Macro Scores')
plt.legend()
plt.grid(True)


plt.savefig("s1_adaboost_acc_f1.png")
plt.show()