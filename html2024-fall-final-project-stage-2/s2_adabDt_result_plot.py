import matplotlib.pyplot as plt


test_idxs = [int(i) for i in range(1,6)]
acc_result = [0.58388, 0.58388, 0.58388, 0.58388, 0.58388]
f1_results = [0.56478, 0.56478, 0.56478, 0.56478, 0.56478]


plt.figure(figsize=(8, 6))
plt.plot(test_idxs, acc_result, marker='o', label='Accuracy')
plt.plot(test_idxs, f1_results, marker='s', label='F1-Macro')

plt.xlabel('Test Index')
plt.ylabel('Score')
plt.title('S2: Accuracy and F1-Macro Scores')
plt.legend()
plt.grid(True)


plt.savefig("s2_adaboost_acc_f1.png")
plt.show()