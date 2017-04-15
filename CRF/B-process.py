import sys
f=open(sys.argv[1],'r')
lines=[]
for line in f:
    lines.append(line)
f.close()
for i in range(len(lines)):
    if lines[i]=='\n':
        continue
    if lines[i].split(' ')[1][0]=='I':
        if lines[i-1]=='\n' or lines[i-1].split(' ')[1]=='O\n' or lines[i-1].split(' ')[1].split('-')[1] != lines[i].split(' ')[1].split('-')[1]:
#            print lines[i-1],lines[i]
            #print lines[i-1].split(' ')[1], lines[i].split(' ')[1]
            lines[i]=lines[i].split(' ')[0]+' '+lines[i].split(' ')[1].replace('I','B')
f=open(sys.argv[1],'w')
for line in lines:
    f.write(line)
