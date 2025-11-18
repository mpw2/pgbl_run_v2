
rfile = open("b1_wall_stats.txt","r")
wfile = open("cf_reference.txt","w")

rfile.readline() # discard header line

for line in rfile:
   stuff = line.strip().split()
   cf = 2.0*(float(stuff[1])**2)
   cfstr = "%.8e" % cf
   wfile.write(stuff[0]+" "+cfstr+"\n")

rfile.close()
wfile.close()
