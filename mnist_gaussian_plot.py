import matplotlib.pyplot as plt
import pickle
scales = [0.40, 0.52, 0.64, 0.76, 0.88, 1.0, 1.12, 1.24, 1.36, 1.48, 1.60]

lists = []
infile = open('mnist_gaussian_log.pickle', 'rb')
while 1:
    try:
        lists.append(pickle.load(infile))
    except (EOFError):
        break
infile.close()

std_test_accs = lists[-1]
std_test_losses = lists[-2]
avg_test_accs = lists[-3]
avg_test_losses = lists[-4]

lists_sr0 = []
infile = open('mnist_gaussian_sr0_log.pickle', 'rb')
while 1:
    try:
        lists_sr0.append(pickle.load(infile))
    except (EOFError):
        break
infile.close()

std_test_accs_sr0 = lists_sr0[-1]
std_test_losses_sr0 = lists_sr0[-2]
avg_test_accs_sr0 = lists_sr0[-3]
avg_test_losses_sr0 = lists_sr0[-4]

kanazawa = [9.082670906200317, 5.104928457869634, 2.7726550079491243, 1.8139904610492845, 1.7853736089030203, 1.4419713831478518, 1.585055643879171, 1.5707472178060407, 2.0715421303656587, 3.0731319554848966, 4.103338632750397]
convnet = [11.286168521462638, 6.449920508744037, 3.4737678855325917, 2.3147853736089026, 2.114467408585055, 1.742448330683624, 1.8426073131955487, 2.128775834658187, 3.1589825119236874, 5.319554848966613, 7.666136724960254]

plt.figure()
plt.errorbar(scales, avg_test_losses, yerr=std_test_losses, label="srange=2")
plt.errorbar(scales, avg_test_losses_sr0, yerr=std_test_losses_sr0, label="srange=0")
plt.title("Average loss vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("test_loss_gaussian_mean_all.pdf")

plt.figure()
plt.errorbar(scales, avg_test_accs, yerr=std_test_accs, label="srange=2")
plt.errorbar(scales, avg_test_accs_sr0, yerr=std_test_accs_sr0, label="srange=0")
plt.title("Average accuracy vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Accuracy %")
plt.legend()
plt.savefig("test_acc_gaussian_mean_all.pdf")

plt.figure()
plt.errorbar(scales, [100-x for x in avg_test_accs], yerr=std_test_accs, label="srange=2")
plt.errorbar(scales, [100-x for x in avg_test_accs_sr0], yerr=std_test_accs_sr0, label="srange=0")
plt.errorbar(scales, kanazawa, label="Kanazawa")
plt.errorbar(scales, convnet, label="ConvNet")
plt.title("Average error vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Error %")
plt.legend()
plt.savefig("test_err_gaussian_mean_all.pdf")