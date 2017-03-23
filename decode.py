import pywrapfst as fst
import sys
import datetime

print "Reading FSTs"
tm = fst.Fst.read(sys.argv[1])
print "Read TM"
lm = fst.Fst.read(sys.argv[2])
print "Read LM"

print "Reading ISYM"
isym = {}
with open(sys.argv[3], "r") as isymfile:
  for line in isymfile:
    x, y = line.strip().split()
    isym[x] = int(y)
print "Read ISYM"


print "Reading OSYM"
osym = {}
with open(sys.argv[4], "r") as osymfile:
  for line in osymfile:
    x, y = line.strip().split()
    osym[int(y)] = x
print "Read OSYM"

#logFile=open("logFile","w")
outFile=open(sys.argv[6],"w")
print "Start Time",datetime.datetime.now()
lineIndex=0
for line in open(sys.argv[5]):
    lineIndex+=1
    #print line
    #logFile.write("Read Input\n")
    # Read input
    compiler = fst.Compiler()
    arr = line.strip().split()
    for i, x in enumerate(arr):
        xsym = isym[x] if x in isym else isym["<unk>"]
        print >> compiler, "%d %d %s %s" % (i, i+1, xsym, xsym)
    print >> compiler, "%s" % (len(arr))
    ifst = compiler.compile()

    #logFile.write("Compiled IFST\n")
    # Create the search graph and do search
    graph = fst.compose(ifst, tm)
    graph = fst.compose(graph, lm)
    graph = fst.shortestpath(graph)

    #logFile.write("Composed FSTs\n")

    # Read off the output
    out = []
    for state in graph.states():
        for arc in graph.arcs(state):
            if arc.olabel != 0 and osym[arc.olabel]!="<unk>":
                out.append(osym[arc.olabel])
    #print(" ".join(reversed(out[1:])))

    outputSentence=" ".join(reversed(out[1:]))
    if lineIndex%10==0:
        print lineIndex
        print outputSentence
        print datetime.datetime.now()

    outFile.write(outputSentence+"\n")
    #logFile.write("Sentence Over")


#logFile.close()
outFile.close()
