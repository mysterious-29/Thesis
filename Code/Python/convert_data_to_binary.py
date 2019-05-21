import csv

with open('WSN-DS_original.csv', 'r') as file1, open('OUTPUT.csv','w',newline='') as file2:
	reader = csv.reader(file1, delimiter=',')
	writer = csv.writer(file2, delimiter=',')
	for row in reader:
		
		if(row[18]=='Normal'):
			replaced = row[18].replace('Normal','1')
			row[18] = replaced
		if(row[18]=='Blackhole'):
			replaced = row[18].replace('Blackhole','0')
			print('Blackhole done')
			row[18] = replaced
		if(row[18]=='Flooding'):
			replaced = row[18].replace('Flooding','0')
			row[18] = replaced
			print('Flooding done')
		if(row[18]=='Grayhole'):
			replaced = row[18].replace('Grayhole','0')
			row[18] = replaced
			print('grayhole done')
		if(row[18]=='TDMA'):
			replaced = row[18].replace('TDMA','0')
			row[18] = replaced
			print('TDMA done')
		# replaced = row[18].replace('Blackhole','0')
		# replaced = row[18].replace('Flooding','0')
		# replaced = row[18].replace('Grayhole','0')
		# replaced = row[18].replace('TDMA','0')
		
		writer.writerow(row)
	print("All Done!")