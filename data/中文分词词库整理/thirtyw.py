# -*- coding: UTF-8 -*- 
f = open('30wChinsesSeqDic.txt')
fout = open('30wdict.txt','a')
count = 0
for line in f:
	temp = line.strip()
	temp_list = temp.split(' ')
	temp_sublist = temp_list[1].split('\t')
	if len(temp_sublist[1]) > 2:
		count = count + 1
		print temp_sublist[1]
		fout.write(temp_sublist[1] + '\n')
f.close()
fout.close()
#print count