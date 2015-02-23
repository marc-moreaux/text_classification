
import csv

train       = '/media/marc/MYLINUXLIVE/data/train.csv'  # path to training file
train_valid = '/media/marc/MYLINUXLIVE/data/train_valid.csv'
train_test  = '/media/marc/MYLINUXLIVE/data/train_test.csv'

label       = '/media/marc/MYLINUXLIVE/data/trainLabels.csv'  # path to label file of training data
label_valid = '/media/marc/MYLINUXLIVE/data/trainLabels_valid.csv'
label_test  = '/media/marc/MYLINUXLIVE/data/trainLabels_test.csv'




csv_valid = open(train_valid,'w')
csv_test = open(train_test,'w')
for t, line in enumerate(open(train)):
	if t > 1700000/5*4:
		csv_test.write(line)
	elif t > 1700000/5*3:
		csv_valid.write(line)

	if not t % 10000:
		print('now at ID %d' % (t))

csv_test.close()
csv_valid.close()


csv_valid = open(label_valid,'w')
csv_test = open(label_test,'w')
for t, line in enumerate(open(label)):
	if t > 1700000/5*4:
		csv_test.write(line)
	elif t > 1700000/5*3:
		csv_valid.write(line)

	if not t % 10000:
		print('now at ID %d' % (t))


csv_test.close()
csv_valid.close()