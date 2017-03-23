from __future__ import print_function
import sys
import math
from collections import defaultdict

def trainTmOneToOne(table,outFileName=None):
    stateid = defaultdict(lambda: len(stateid))

    with open(outFileName, "w") as outfile:
        for word_en in table:
            wordStr=word_en
            if wordStr=="<s>":
                wordStr="<eps>"
            for word_fr in table[word_en]:
                v1=table[word_en][word_fr]
                print("%d %d %s %s %.4f" % (stateid["<s>"], stateid["<s>"], word_fr, wordStr, -math.log(v1)), file=outfile)
    
        print(stateid["<s>"], file=outfile) 

def trainTmSmart(table,outFileName=None):
    stateid = defaultdict(lambda: len(stateid))
    FERTILITY=3

    with open(outFileName, "w") as outfile:
        for word_en in table:
            wordStr=word_en
            if wordStr=="<s>":
                wordStr="<eps>"

            
            print("%d %d %s %s %.4f" % (stateid[word_en], stateid["<s>"],"<eps>", "<eps>", 0.0), file=outfile)
            print("%d %d %s %s %.4f" % (stateid[word_en+"_"+str(0)], stateid["<s>"],"<eps>", "<eps>", 0.0), file=outfile)

            for word_fr in table[word_en]:
                v1=table[word_en][word_fr]
                print("%d %d %s %s %.4f" % (stateid["<s>"], stateid[word_en],word_fr, wordStr,-math.log(v1)), file=outfile)
                print("%d %d %s %s %.4f" % (stateid[word_en], stateid[word_en+"_"+str(0)],word_fr,"<eps>",-3*math.log(v1)), file=outfile)


        print(stateid["<s>"], file=outfile) 


def parse(phrase):
    li=[]
    for x in phrase:
        if x=="<s>":
            x="<eps>"
        li.append(x)

    return tuple(li)

def compose(tup1,tup2):
    return tuple(list(tup1)+list(tup2))

def trainTmPhrase(table,table2,outFileName=None,useTable2=True):
    stateid = defaultdict(lambda: len(stateid))

    with open(outFileName, "w") as outfile:
        for phrase_en in table:
            phraseStr=parse(phrase_en)
            for phrase_fr in table[phrase_en]:
                v1=table[phrase_en][phrase_fr]
                 
                print("%d %d %s %s %.4f" % (stateid[""], stateid[phrase_fr[:1]], phrase_fr[0],"<eps>",0.0), file=outfile) 
                for i in range(1,len(phrase_fr)):
                    print("%d %d %s %s %.4f" % (stateid[phrase_fr[:i]], stateid[phrase_fr[:i+1]], phrase_fr[i],"<eps>",0.0), file=outfile)

                print("%d %d %s %s %.4f" % (stateid[phrase_fr], stateid[phrase_fr+("<sep>",)+phraseStr[:1]],"<eps>",phraseStr[0],0.0), file=outfile) 
                for i in range(1,len(phraseStr)):
                    print("%d %d %s %s %.4f" % (stateid[phrase_fr+("<sep>",)+phraseStr[:i]], stateid[phrase_fr+("<sep>",)+phraseStr[:i+1]],"<eps>",phraseStr[i],0.0), file=outfile) 
                print("%d %d %s %s %.4f" % (stateid[phrase_fr+("<sep>",)+phraseStr], stateid[""],"<eps>","<eps>",-math.log(v1)), file=outfile) 

        if useTable2:
            table=table2
            for word_en in table:
                wordStr=word_en
                if wordStr=="<s>":
                    wordStr="<eps>"
                for word_fr in table[word_en]:
                    v1=table[word_en][word_fr]
                    print("%d %d %s %s %.4f" % (stateid[""], stateid[""], word_fr, wordStr, -math.log(v1)), file=outfile)
     
        print(stateid[""], file=outfile) 


def trainTmPhrase2(table,table2,outFileName=None,useTable2=True):
    stateid = defaultdict(lambda: len(stateid))
    
    #print "Reversing Tables"
    reverseTable={}
    for phrase_en in table:
        for phrase_fr in table[phrase_en]:
            if phrase_fr not in reverseTable:
                reverseTable[phrase_fr]={}
            reverseTable[phrase_fr][phrase_en]=table[phrase_en][phrase_fr]
    table=None
    #print "Reversed Tables"

    with open(outFileName, "w") as outfile:
        for phrase_fr in reverseTable:
            print("%d %d %s %s %.4f" % (stateid[""], stateid[phrase_fr[:1]], phrase_fr[0],"<eps>",0.0), file=outfile) 
            for i in range(1,len(phrase_fr)):
                print("%d %d %s %s %.4f" % (stateid[phrase_fr[:i]], stateid[phrase_fr[:i+1]], phrase_fr[i],"<eps>",0.0), file=outfile)

            for phrase_en in reverseTable[phrase_fr]:
                phraseStr=parse(phrase_en)
                v1=reverseTable[phrase_fr][phrase_en]
                 
                print("%d %d %s %s %.4f" % (stateid[phrase_fr], stateid[phrase_fr+("<sep>",)+phraseStr[:1]],"<eps>",phraseStr[0],0.0), file=outfile) 
                for i in range(1,len(phraseStr)):
                    print("%d %d %s %s %.4f" % (stateid[phrase_fr+("<sep>",)+phraseStr[:i]], stateid[phrase_fr+("<sep>",)+phraseStr[:i+1]],"<eps>",phraseStr[i],0.0), file=outfile) 
                print("%d %d %s %s %.4f" % (stateid[phrase_fr+("<sep>",)+phraseStr], stateid[""],"<eps>","<eps>",-math.log(v1)), file=outfile) 

        if useTable2:
            table=table2
            for word_en in table:
                wordStr=word_en
                if wordStr=="<s>":
                    wordStr="<eps>"
                for word_fr in table[word_en]:
                    v1=table[word_en][word_fr]
                    print("%d %d %s %s %.4f" % (stateid[""], stateid[""], word_fr, wordStr, -math.log(v1)), file=outfile)
     
        print(stateid[""], file=outfile) 

def trainTmPhrase3(table,table2,outFileName=None,useTable2=True):
    stateid = defaultdict(lambda: len(stateid))
    
    #print "Reversing Tables"
    reverseTable={}
    for phrase_en in table:
        for phrase_fr in table[phrase_en]:
            if phrase_fr not in reverseTable:
                reverseTable[phrase_fr]={}
            reverseTable[phrase_fr][phrase_en]=table[phrase_en][phrase_fr]
    table=None
    #print "Reversed Tables"

    with open(outFileName, "w") as outfile:
        for phrase_fr in reverseTable:
            print("%d %d %s %s %.4f" % (stateid[""], stateid[phrase_fr[:1]], phrase_fr[0],"<eps>",0.0), file=outfile) 
            for i in range(1,len(phrase_fr)):
                print("%d %d %s %s %.4f" % (stateid[phrase_fr[:i]], stateid[phrase_fr[:i+1]], phrase_fr[i],"<eps>",0.0), file=outfile)

            for phrase_en in reverseTable[phrase_fr]:
                phraseStr=parse(phrase_en)
                v1=reverseTable[phrase_fr][phrase_en]
                 
                print("%d %d %s %s %.4f" % (stateid[phrase_fr], stateid[phrase_fr+("<sep>",)+phraseStr[:1]],"<eps>",phraseStr[0],0.0), file=outfile) 
                for i in range(1,len(phraseStr)):
                    print("%d %d %s %s %.4f" % (stateid[phrase_fr+("<sep>",)+phraseStr[:i]], stateid[phrase_fr+("<sep>",)+phraseStr[:i+1]],"<eps>",phraseStr[i],0.0), file=outfile) 
                print("%d %d %s %s %.4f" % (stateid[phrase_fr+("<sep>",)+phraseStr], stateid[""],"<eps>","<eps>",-math.log(v1)), file=outfile) 

        if useTable2:
            table=table2
            for word_en in table:
                wordStr=word_en
                if wordStr=="<s>":
                    wordStr="<eps>"
                print("%d %d %s %s %.4f" % (stateid[word_en], stateid[""],"<eps>", "<eps>", 0.0), file=outfile)
                print("%d %d %s %s %.4f" % (stateid[word_en+"_"+str(0)], stateid[""],"<eps>", "<eps>", 0.0), file=outfile)


                for word_fr in table[word_en]:
                    v1=table[word_en][word_fr]
                    print("%d %d %s %s %.4f" % (stateid[""], stateid[word_en], word_fr, wordStr, -math.log(v1)), file=outfile)
                    print("%d %d %s %s %.4f" % (stateid[word_en], stateid[word_en+"_"+str(0)],word_fr,"<eps>",-1.5*math.log(v1)), file=outfile)


        print(stateid[""], file=outfile) 



def trainTmPhraseNew(table,table2,outFileName=None,useTable2=True):
    stateid = defaultdict(lambda: len(stateid))

    with open(outFileName, "w") as outfile:
        for phrase_en in table:
            phraseStr=parse(phrase_en)
            
            print("%d %d %s %s %.4f" % (stateid[""], stateid[phraseStr[:1]],"<eps>",phraseStr[0],0.0), file=outfile) 
            for i in range(1,len(phraseStr)):
                print("%d %d %s %s %.4f" % (stateid[phraseStr[:i]], stateid[phraseStr[:i+1]],"<eps>",phraseStr[i],0.0), file=outfile) 
 
            for phrase_fr in table[phrase_en]:
                v1=table[phrase_en][phrase_fr]
                 
                print("%d %d %s %s %.4f" % (stateid[phraseStr], stateid[phraseStr+("<sep>",)+phrase_fr[:1]], phrase_fr[0],"<eps>",0.0), file=outfile) 
                for i in range(1,len(phrase_fr)):
                    print("%d %d %s %s %.4f" % (stateid[phraseStr+("<sep>",)+phrase_fr[:i]], stateid[phraseStr+("<sep>",)+phrase_fr[:i+1]], phrase_fr[i],"<eps>",0.0), file=outfile)

                print("%d %d %s %s %.4f" % (stateid[phraseStr+("<sep>",)+phrase_fr], stateid[""],"<eps>","<eps>",-math.log(v1)), file=outfile) 

        if useTable2:
            table=table2
            for word_en in table:
                wordStr=word_en
                if wordStr=="<s>":
                    wordStr="<eps>"
                for word_fr in table[word_en]:
                    v1=table[word_en][word_fr]
                    print("%d %d %s %s %.4f" % (stateid[""], stateid[""], word_fr, wordStr, -math.log(v1)), file=outfile)
     
        print(stateid[""], file=outfile) 




def trainSymbolic(sentences,outFileName=None):
    ctxts1 = 0.0
    ctxts2 = defaultdict(lambda: 0.0)
    count1 = defaultdict(lambda: 0.0)
    count2 = defaultdict(lambda: 0.0)
    for sentence in sentences:
        vals = sentence
        ctxt = "<s>"
        for val in vals:
          ctxts1 += 1
          ctxts2[ctxt] += 1
          count1[val] += 1
          count2[(ctxt,val)] += 1
          ctxt = val

    ALPHA_1 = 0.1
    ALPHA_2 = 1.0 - ALPHA_1

    stateid = defaultdict(lambda: len(stateid))

    with open(outFileName, "w") as outfile:

      # Print the fallbacks
      print("%d %d <eps> <eps> %.4f" % (stateid["<s>"], stateid[""], -math.log(ALPHA_1)), file=outfile)
      for ctxt, val in ctxts2.items():
        if ctxt != "<s>":
          print("%d %d <eps> <eps> %.4f" % (stateid[ctxt], stateid[""], -math.log(ALPHA_1)), file=outfile)
      
      # Print the unigrams
      for word, val in count1.items():
        v1 = val/ctxts1
        print("%d %d %s %s %.4f" % (stateid[""], stateid[word], word, word, -math.log(v1)), file=outfile)
      
      # Print the unigrams
      for (ctxt, word), val in count2.items():
        v1 = count1[word]/ctxts1
        v2 = val/ctxts2[ctxt]
        val = ALPHA_2 * v2 + ALPHA_1 * v1
        print("%d %d %s %s %.4f" % (stateid[ctxt], stateid[word], word, word, -math.log(val)), file=outfile)
      
      # Print the final state
      print(stateid["</s>"], file=outfile) 
      
