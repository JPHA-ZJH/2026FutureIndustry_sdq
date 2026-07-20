import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar([1,2,3], [10,20,30])
fig.savefig('outputs/figures/test_fig.png')
print('Test figure saved')
