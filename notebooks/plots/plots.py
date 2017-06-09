import matplotlib.pyplot as plt

epochs = range(1, 6)

# accuracy
# training = [0.20, 0.20, 0.20, 0.20, 0.20]
# validation = [0.18, 0.18, 0.18, 0.18, 0.18]
#
# plt.plot(epochs, training, 'r', label="training")
# plt.plot(epochs, validation, 'b', label="validation")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend(loc="upper left")
# plt.savefig("plot.png")


# loss
# training = [36.469, 31.919, 31.569, 31.448, 31.375, ]
# validation = [81.896, 80.436, 80.027, 80.195, 80.031]
#
# plt.plot(epochs, training, 'r', label="training")
# plt.plot(epochs, validation, 'b', label="validation")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend(loc="upper left")
# plt.savefig("plot.png")


# AER
aer = [0.47, 0.46, 0.46, 0.45, 0.45]
plt.plot(epochs, aer, 'r', label="validation")
plt.xlabel("Epochs")
plt.ylabel("AER")
plt.legend(loc="upper left")
plt.savefig("plot.png")