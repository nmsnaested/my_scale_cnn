import matplotlib.pyplot as plt
import pickle

scales = [(1.0, 1.0), (0.9, 1.1), (0.8, 1.2), (0.6, 1.4), (0.5, 1.5), (0.4, 1.6), (0.3, 1.7)]

lists = []
infile = open('cifar_range_log.pickle', 'rb')
while 1:
    try:
        lists.append(pickle.load(infile))
    except (EOFError):
        break
infile.close()


std_test_accs = lists[-1]
avg_test_accs = lists[-2]
std_test_losses = lists[-3]
avg_test_losses = lists[-4]

lists_sr0 = []
infile = open('cifar_range_sr0_log.pickle', 'rb')
while 1:
    try:
        lists_sr0.append(pickle.load(infile))
    except (EOFError):
        break
infile.close()

std_test_accs_sr0 = lists_sr0[-1]
avg_test_accs_sr0 = lists_sr0[-2]
std_test_losses_sr0 = lists_sr0[-3]
avg_test_losses_sr0 = lists_sr0[-4]

plt.figure()
plt.errorbar([str(s) for s in scales], avg_test_losses, yerr=std_test_losses, label="SiCNN_3 srange=0")
plt.errorbar([str(s) for s in scales], avg_test_losses_sr0, yerr=std_test_losses_sr0, label="SiCNN_3 srange=2")
plt.title("Average loss vs Scale factor")
plt.xlabel("Scale range")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("cifar_test_loss_range_mean.pdf")

plt.figure()
plt.errorbar([str(s) for s in scales], avg_test_accs, yerr=std_test_accs, label="SiCNN_3 srange=0")
plt.errorbar([str(s) for s in scales], avg_test_accs_sr0, yerr=std_test_accs_sr0, label="SiCNN_3 srange=2")
plt.title("Average accuracy vs Scale factor")
plt.xlabel("Scale range")
plt.ylabel("Accuracy %")
plt.legend()
plt.savefig("cifar_test_acc_range_mean.pdf")

plt.figure()
plt.errorbar([str(s) for s in scales], [100-x for x in avg_test_accs], yerr=std_test_accs, label="SiCNN_3 srange=0")
plt.errorbar([str(s) for s in scales], [100-x for x in avg_test_accs_sr0], yerr=std_test_accs_sr0, label="SiCNN_3 srange=2")
plt.title("Average error vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Error %")
plt.legend()
plt.savefig("cifar_test_err_range_mean.pdf")
"""
################################################################

scales = [0.40, 0.52, 0.64, 0.76, 0.88, 1.0, 1.12, 1.24, 1.36, 1.48, 1.60]

infile = open("cifar_gaussian_log_results.pickle", "rb")

glist = []
while 1:
    try:
        glist.append(pickle.load(infile))
    except (EOFError):
        break


for i, m in enumerate(range(6,-1,-1)):
    locals()['avg_test_loss_{0}'.format(m)] = glist[-(4+i*4)]
    locals()['avg_test_acc_{0}'.format(m)] = glist[-(3+i*4)]
    locals()['std_test_loss_{0}'.format(m)] = glist[-(2+i*4)]
    locals()['std_test_acc_{0}'.format(m)] = glist[-(1+i*4)]

infile.close()

plt.figure()
for m in range(7): 
    plt.errorbar(scales, locals()['avg_test_loss_{0}'.format(m)], yerr=locals()['std_test_loss_{0}'.format(m)], label="model {}".format(m))
plt.title("Mean Loss vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("avg_test_loss_gaussian_cifar.pdf")

plt.figure()
for m in range(7): 
    plt.errorbar(scales, locals()['avg_test_acc_{0}'.format(m)], yerr=locals()['std_test_acc_{0}'.format(m)], label="model {}".format(m))
plt.title("Mean Accuracy vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Accuracy %")
plt.legend()
plt.savefig("avg_test_acc_gaussian_cifar.pdf")

for m in range(7): 
    locals()['avg_test_err_{0}'.format(m)] = [100-l for l in locals()['avg_test_acc_{0}'.format(m)]]

plt.figure()
for m in range(7): 
    plt.errorbar(scales, locals()['avg_test_err_{0}'.format(m)], yerr=locals()['std_test_acc_{0}'.format(m)], label="model {}".format(m))
plt.title("Mean Error vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Error %")
plt.legend()
plt.savefig("avg_test_err_gaussian_cifar.pdf")
"""