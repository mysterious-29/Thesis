import csv

with open('WSN-DS_original.csv', 'r') as file1, open('black-hole.csv','w',newline='') as file2:
	reader = csv.reader(file1, delimiter=',')
	writer = csv.writer(file2, delimiter=',')
	for row in reader:
		if(row[18]=='Normal' or row[18]=='Blackhole'):			
			writer.writerow(row)
	print("All Done!")