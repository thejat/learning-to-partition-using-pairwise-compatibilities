fig = plt.figure()
ax = fig.add_subplot(111)

temp = numpy.asarray(performance)
#print temp
temp_avg = numpy.mean(temp, axis=0)
#print temp_avg
temp_std = numpy.std(temp, axis=0)

xs = [10,100,200,500,1000]
ys = temp_avg

ax.fill_between(xs, ys-temp_std, ys+temp_std, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
ax.plot(xs, ys)

plt.show()
plt.xlabel('Sample size #')
plt.ylabel('Normalized rror')
plt.title('Error between estimates and true scores')
plt.tight_layout()
plt.ylim((0,1.2*max(ys)))

plot_dir = '.'
fname = plot_dir + '/scores_fig' + '.png'
print "plot will be saved at:",fname
plt.savefig(fname)