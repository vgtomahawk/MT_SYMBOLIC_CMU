from collections import defaultdict
import readData
import utilities
import pickle
import datetime
import math
import pywrapfst as fst

class Config:
    def __init__(self,EM_EPOCHS=25,MIN_FREQUENCY_DE=3,MIN_FREQUENCY_EN=5,ALPHA_1=0.1,ALPHA_UNK=0.01,EPS=0,UNK=1,STOP=2,START=3,NULL=4,SEPARATOR=5,TINIT=6):
        self.EM_EPOCHS=EM_EPOCHS
        self.MIN_FREQUENCY_DE=MIN_FREQUENCY_DE
        self.MIN_FREQUENCY_EN=MIN_FREQUENCY_EN
        self.ALPHA_1=ALPHA_1
        self.ALPHA_UNK=ALPHA_UNK
        self.ALPHA_2=1.0-self.ALPHA_1-self.ALPHA_UNK
        self.PROB_UNK=ALPHA_UNK/10000000
        self.EPS=EPS
        self.UNK=UNK
        self.STOP=STOP
        self.START=START
        self.NULL=NULL
        self.SEPARATOR=SEPARATOR
        self.TINIT=TINIT

class Model1:

    def __init__(self,config=None):
        self.config=config
        self.wids_en=defaultdict(int)
        self.wids_de=defaultdict(int)
        
        print "Reading Data"
        self.train_sentences_de=readData.read_corpus(self.wids_de,mode="train",update_dict=True,min_frequency=self.config.MIN_FREQUENCY_DE,language="de")
        self.train_sentences_en=readData.read_corpus(self.wids_en,mode="train",update_dict=True,min_frequency=self.config.MIN_FREQUENCY_EN,language="en")
        print "Finished Reading Data"
       
        self.reverse_wids_en=utilities.reverseDict(self.wids_en)
        self.reverse_wids_de=utilities.reverseDict(self.wids_de)

        print "German Dictionary Size",len(self.wids_de)
        print "English Dictionary Size",len(self.wids_en)

        print "Initializing Sparse Count Tables"
        #Sparse dictionary style, maintains P(f_j|e_j)
        self.probs={}

        self.train_sentences=zip(self.train_sentences_de,self.train_sentences_en)
        self.train_sentences_de=None
        self.train_sentences_en=None

        #Init probs
        for train_sentence in self.train_sentences:
            train_sentence_de=train_sentence[0]
            train_sentence_en=[self.config.START,]+train_sentence[1]
            for word_en in train_sentence_en:
                if word_en not in self.probs:
                    self.probs[word_en]={}
                for word_de in train_sentence_de:
                    if word_de not in self.probs[word_en]:
                        self.probs[word_en][word_de]=1.0

        #Normalize probs
        for word_en in self.probs:
            Z=len(self.probs[word_en])
            for word_de in self.probs[word_en]:
                self.probs[word_en][word_de]=1.0/(Z+0.0)


        self.phraseProbs={}
    
    def readValidationData(self):
        print "Reading Validation Data"
        self.valid_sentences_de=readData.read_corpus(self.wids_de,mode="valid",update_dict=False,min_frequency=self.config.MIN_FREQUENCY_DE,language="de")
        self.valid_sentences_en=readData.read_corpus(self.wids_en,mode="valid",update_dict=False,min_frequency=self.config.MIN_FREQUENCY_EN,language="en")
        print "Finished Reading Validation Data"
        self.valid_sentences=zip(self.valid_sentences_de,self.valid_sentences_en)
        self.valid_sentences_de=None
        self.valid_sentences_en=None

    def convertAndWriteSentences(self,mode):
        print "Reading Data"
        valid_sentences_de=readData.read_corpus(self.wids_de,mode=mode,update_dict=False,min_frequency=self.config.MIN_FREQUENCY_DE,language="de")
        print "Finished Reading Data"
        outFile=open(mode+".convert","w")
        for sentence in valid_sentences_de:
            convertedSentence=" ".join([self.reverse_wids_de[x] for x in sentence])
            outFile.write(convertedSentence+"\n")
        outFile.close()


    def trainNGramModel(self):
        print "Training NGram Model"
        ctxts1=0.0
        ctxts2=defaultdict(lambda: 0.0)
        count1=defaultdict(lambda: 0.0)
        count2=defaultdict(lambda: 0.0)
        for train_sentence in self.train_sentences:
            train_sentence_en=train_sentence[1]
            ctxt=self.config.START
            for val in train_sentence_en:
                ctxts1+=1
                ctxts2[ctxt]+=1
                count1[val]+=1
                count2[(ctxt,val)]+=1
                ctxt=val
        
        self.ctxts1=ctxts1
        self.ctxts2=ctxts2
        self.count1=count1
        self.count2=count2

        print "Finished Training NGram Model"

    def writeLanguageFST(self):
        self.stateId=defaultdict(lambda: len(self.stateId))
        stateId=self.stateId
        fp=open("lm.fst","w")

        fp.write("%d %d %d %d %.4f\n" % (stateId[self.config.START],stateId[-1],-math.log(self.config.ALPHA_1)) )
        
        for ctxt,val in self.ctxts2.items():
            if ctxt!=self.config.START:
                fp.write("%d %d %d %d %.4f\n" % (stateId[ctxt],stateId[-1],-math.log(self.config.ALPHA_1)) )
       
        for word,val in self.count1.items():
            v1=val/self.ctxts1
            fp.write("%d %d %d %d %.4f\n" % (stateId[-1],stateId[word],word,word,-math.log(v1)) )

        for (ctxt,word),val in self.count2.items():
            v1=self.count1[word]/self.ctxts1
            v2=val/self.ctxts2[ctxt]
            val=(1-self.config.ALPHA_1)*v2+self.config.ALPHA_1*v1
            fp.write("%d %d %d %d %.4f\n" % (stateId[ctxt],stateId[word],word,word,-math.log(v1)) )


        fp.write(str(stateId[self.config.STOP])+"\n")
        fp.close()

    def writeLanguageFSTOnline(self):
        self.stateId=defaultdict(lambda: len(self.stateId))
        stateId=self.stateId
        compiler=fst.Compiler()

        print >> compiler, "%d %d %d %d %.4f" % (stateId[self.config.START],stateId[self.config.NULL],self.config.GARBAGE,self.config.GARBAGE,-math.log(self.config.ALPHA_1))
          
        for ctxt,val in self.ctxts2.items():
            if ctxt!=self.config.START:
                print>> compiler, "%d %d %d %d %.4f" % (stateId[ctxt],stateId[self.config.NULL],self.config.GARBAGE,self.config.GARBAGE,-math.log(self.config.ALPHA_1))
       
        for word,val in self.count1.items():
            v1=val/self.ctxts1
            print>> compiler, "%d %d %d %d %.4f" % (stateId[self.config.NULL],stateId[word],word,word,-math.log(v1))

        for (ctxt,word),val in self.count2.items():
            v1=self.count1[word]/self.ctxts1
            v2=val/self.ctxts2[ctxt]
            val=(1-self.config.ALPHA_1)*v2+self.config.ALPHA_1*v1
            print>> compiler, "%d %d %d %d %.4f" % (stateId[ctxt],stateId[word],word,word,-math.log(val))


        print>> compiler, str(stateId[self.config.STOP])
        
        self.lmfst=compiler.compile()
        self.lmfst.arcsort()

        print "Finished Creating LM FST"
        print "LM FST has",self.lmfst.num_states(),"States"
        print "LM FST has",self.lmfst.start(),"Start State"

    def cTup(self,e1,e2):
        return tuple(list(e1)+list(e2))

    def writeTranslationFSTOnline(self):
        compiler=fst.Compiler()
        self.tStateId=defaultdict(lambda: len(self.tStateId))
        stateId=self.tStateId
        
        uniqueState=stateId[self.config.TINIT]
        for ep in self.phraseProbs:
            for fp,val in self.phraseProbs[ep].items():
                print>> compiler, "%d %d %d %d %.4f" % (uniqueState,stateId[fp[:1]],fp[0],self.config.GARBAGE,0.0)
                for i in range(1,len(fp)):
                    print>> compiler, "%d %d %d %d %.4f" % (stateId[fp[:i]],stateId[fp[:(i+1)]],fp[i],self.config.GARBAGE,0.0)
                
                prevRep=fp[:len(fp)]
                prevState=stateId[prevRep]
                finalRep=tuple(list(fp[:len(fp)])+[self.config.SEPARATOR,])
                finalState=stateId[finalRep]
                print>> compiler, "%d %d %d %d %.4f" % (prevState,finalState,self.config.GARBAGE,self.config.GARBAGE,0.0)

                
                print>> compiler, "%d %d %d %d %.4f" % (finalState,stateId[self.cTup(finalRep,ep[:1])],self.config.GARBAGE,ep[0],0.0)

                for i in range(1,len(ep)):
                    print>> compiler, "%d %d %d %d %.4f"  % (stateId[self.cTup(finalRep,ep[:i])],stateId[self.cTup(finalRep,ep[:(i+1)])],self.config.GARBAGE,ep[i],0.0)
                
                completeState=stateId[self.cTup(finalRep,ep[:len(ep)])]
                print>> compiler, "%d %d %d %d %.4f" % (completeState,uniqueState,self.config.GARBAGE,self.config.GARBAGE,-math.log(val))

        print>> compiler, str(uniqueState)
        
        self.tmfst=compiler.compile()
        self.tmfst.arcsort()

        print "Finished Creating TM FST"
        print "TM FST has",self.tmfst.num_states(),"States"

    def getOut(self,inputFST):
        graph=fst.shortestpath(inputFST)
        out=[]
        for state in graph.states():
            for arc in graph.arcs(state):
                out.append(arc.olabel)
        return out

    def writeIFSTOnline(self,sentence_de):
        maxId=len(self.stateId)+1
        compiler=fst.Compiler()
        for i,x in enumerate(sentence_de):
            print>> compiler, "%d %d %d %d" % (i,i+1,x,x)
        print>> compiler, "%d" % (maxId+len(sentence_de))
        ifst=compiler.compile()  
        return ifst

    def decode(self,sentence_de):
        #ifst=self.writeIFSTOnline(sentence_de)

        print "LM States",self.lmfst.num_states()
        #print "TM States",self.tmfst.num_states()

        #Find most likely sentence in the LM
        for state in self.lmfst.states():
            print state,self.lmfst.final(state)

        exit()

    def generateValidationOutput(self):
        for sentence in self.valid_sentences:
            validation_sentence_de=sentence[0]
            self.decode(validation_sentence_de)

    def extractAlignment(self,sentence,verbose=False):
        sentence_de=sentence[0]
        sentence_en=[self.config.START,]+sentence[1]
        #German position -> English position
        alignments={}
        #English positions to German position
        reverseAlignments={}
        reverseAlignments[0]=set()
        totalAlignmentScore=0.0
        for i,word_de in enumerate(sentence_de):
            maxAlignment=-1
            maxAlignmentScore=-float("inf")
            
            for j,word_en in enumerate(sentence_en):
                alignmentScore=-float("inf")
                try:
                    alignmentScore=self.probs[word_en][word_de]
                except KeyError:
                    alignmentScore=-float("inf")
                if alignmentScore>maxAlignmentScore:
                    maxAlignment=j
                    maxAlignmentScore=alignmentScore
                elif alignmentScore==maxAlignmentScore and abs(maxAlignment-i)>abs(j-i):
                    maxAlignment=j
            
            totalAlignmentScore+=math.log(maxAlignmentScore)
            alignments[i]=maxAlignment
            if maxAlignment not in reverseAlignments:
                reverseAlignments[maxAlignment]=set()
            reverseAlignments[maxAlignment].add(i)
        #Returned alignments are German to English word
        
        totalAlignmentProb=math.exp(totalAlignmentScore)

        if verbose:
            for i,word_de in enumerate(sentence_de):
                print self.reverse_wids_de[sentence_de[i]],self.reverse_wids_en[sentence_en[alignments[i]]]

        return alignments,reverseAlignments,totalAlignmentProb

    def materializePhrase(self,sentence,tup):
        sentence_de=sentence[0]
        sentence_en=[self.config.START,]+sentence[1]
        
        i1=tup[0]
        i2=tup[1]
        j1=tup[2]
        j2=tup[3]

        return (tuple(sentence_en[i1:i2+1]),tuple(sentence_de[j1:j2+1]))

    def trainPhrases(self,trainOn="train",verbose=False,writeToFile=True,fileName="phraseProbs.txt",enPhraseLimit=3,frPhraseLimit=3,probabilistic=True):
        print "Extracting Phrases"
        self.phraseCounts={}
        if trainOn=="train":
            sentences=self.train_sentences
        elif trainOn=="valid":
            sentences=self.valid_sentences

        for i,sentence in enumerate(sentences):
            if i%20000==0:
                print "Finished Sentence",i
            matBP,totalProb=self.extractPhrases(sentence,enPhraseLimit=enPhraseLimit,frPhraseLimit=frPhraseLimit)
            for bp in matBP:
                if bp[0] not in self.phraseCounts:
                    self.phraseCounts[bp[0]]={}
                if bp[1] not in self.phraseCounts[bp[0]]:
                    self.phraseCounts[bp[0]][bp[1]]=0.0
                if probabilistic==False:
                    self.phraseCounts[bp[0]][bp[1]]+=1.0
                elif probabilistic==True:
                    #print totalProb
                    self.phraseCounts[bp[0]][bp[1]]+=totalProb*1e4
                   
        #Normalize
        toBeDeleted=[]
        for ep,dic in self.phraseCounts.items():
            Z=sum([dic[fp] for fp in dic])
            if Z==0.0:
                toBeDeleted.append(ep)
                continue
            for fp in self.phraseCounts[ep]:
                self.phraseCounts[ep][fp]/=Z

        for key in toBeDeleted:
            self.phraseCounts.pop(key,None)

        self.phraseProbs=self.phraseCounts
        self.phraseCounts=None
        print "Finished Extracting Phrases"

        if writeToFile:
            filePointer=open(fileName,"w")
            for ep in self.phraseProbs:
                fpList=[(fp,val) for fp,val in self.phraseProbs[ep].items()]
                fpList.sort(key=lambda x:-x[1])
                for fp,val in fpList:
                    englishPhrase=" ".join(list([self.reverse_wids_en[c] for c in ep]))
                    frenchPhrase=" ".join(list([self.reverse_wids_de[c] for c in fp]))
                    filePointer.write(englishPhrase+"\t"+frenchPhrase+"\t"+str(val)+"\n")
            filePointer.close()

    def extractPhrases(self,sentence,verbose=False,enPhraseLimit=3,frPhraseLimit=3):
        alignments,reverseAlignments,totalAlignmentProb=self.extractAlignment(sentence)
        sentence_de=sentence[0]
        sentence_en=[self.config.START,]+sentence[1]
        BP=set()

        for i1 in range(1,len(sentence_en)):
            for i2 in range(1,len(sentence_en)):
                if i2>=i1+enPhraseLimit:
                    continue
                TP=set()
                for i in range(i1,i2+1):
                    if i in reverseAlignments:
                        TP=TP.union(reverseAlignments[i])
                if len(TP)==0:
                    continue

                TP=list(TP)
                TP.sort()
                nullAlignedIndices=reverseAlignments[0]
                quasiConsecutive=True
                j1=TP[0]
                j2=TP[-1]
                #Every index should in closed interval j1-j2 should either be in TP or null-aligned
                for j in range(j1,j2+1):
                    if j in nullAlignedIndices:
                        continue
                    if j not in TP:
                        quasiConsecutive=False
                        break
                if quasiConsecutive:
                    SP=set()
                    for j in range(j1,j2+1):
                        if alignments[j]!=0:
                            SP.add(alignments[j])
                    SP=list(SP)
                    SP.sort()

                    if SP[0]>=i1 and SP[-1]<=i2:
                        if j2<j1+frPhraseLimit:        
                            BP.add((i1,i2,j1,j2))

                        j1=j1-1
                        while j1>=0 and j1 in nullAlignedIndices:
                            jprime=j2+1
                            while jprime<len(sentence_de) and jprime in nullAlignedIndices:
                                if jprime<j1+frPhraseLimit:
                                    BP.add((i1,i2,j1,jprime))
                                jprime=jprime+1
                            j1=j1-1
                


        matBP=[]
        for bp in BP:
            matBP.append(self.materializePhrase(sentence,bp))
       
        if verbose:
            for bp in matBP:
                englishPhrase=[self.reverse_wids_en[i] for i in bp[0]]
                frenchPhrase=[self.reverse_wids_de[i] for i in bp[1]]
                print "English Phrase",englishPhrase,"French Phrase",frenchPhrase

        return matBP,totalAlignmentProb

    def writeProbsToFile(self,outFileName="alignmentProbs.txt"):
        outFile=open(outFileName,"w")

        for word_en in self.probs:
            scoreList=[(key,value) for key,value in self.probs[word_en].items()]
            scoreList.sort(key=lambda x: -x[1])
            for key,value in scoreList:
                outFile.write(self.reverse_wids_en[word_en]+"\t"+self.reverse_wids_de[key]+"\t"+str(value)+"\n")


        outFile.close()

    def saveModel(self,outFileName="model.p"):
        print "Saving Model"
        pickle.dump(self.probs,open(outFileName,"wb"))
        print "Saved Model"

    def loadModel(self,inFileName="model.p"):
        print "Loading Model"
        self.probs=pickle.load(open(inFileName,"rb"))
        print "Loaded Model"

                       

    def train(self):
        

        for epochId in range(self.config.EM_EPOCHS):
            countTable={}
            countZ={}

            for word_en in self.probs:
                countTable[word_en]={}
                countZ[word_en]=0.0

                for word_de in self.probs[word_en]:
                    countTable[word_en][word_de]=0.0

            print "Starting E-Step"
            #E-STEP
            for trainId,train_sentence in enumerate(self.train_sentences):
                if trainId%20000==0:
                    print "Sentence Id",trainId

                train_sentence_de=train_sentence[0]
                train_sentence_en=[self.config.START,]+train_sentence[1]
        
                for word_de in train_sentence_de:
                    Z=0.0
                    for t in train_sentence_en:
                        Z+=self.probs[t][word_de]

                    for t in train_sentence_en:
                        p_t=self.probs[t][word_de]/Z
                        countTable[t][word_de]+=p_t
                        countZ[t]+=p_t

            print "Starting With M-Step"
            #M-Step
            self.probs={}
            for word_en in countZ:
                self.probs[word_en]={}
                for word_de in countTable[word_en]:
                    self.probs[word_en][word_de]=countTable[word_en][word_de]/countZ[word_en]

            print "Done with Epoch",epochId

    def trainNGramSymbolic(self):
        train_sentences_en=[x[1] for x in self.train_sentences]
        reverse_train_sentences_en=[[self.reverse_wids_en[x] for x in train_sentence_en] for train_sentence_en in train_sentences_en]
        import trainngram
        trainngram.trainSymbolic(reverse_train_sentences_en,outFileName="ngram-fst.txt")

    def trainTmSymbolic(self,type="OneToOne"):
        table={}
        for word_en in self.probs:
            eKey=self.reverse_wids_en[word_en]
            table[eKey]={}
            topValues=[(word_de,val) for word_de,val in self.probs[word_en].items()]
            topValues.sort(key = lambda x:-x[1])
            for word_de,val in topValues[:10]:
                fKey=self.reverse_wids_de[word_de]
                table[eKey][fKey]=val
        import trainngram
        if type=="OneToOne":
            trainngram.trainTmOneToOne(table,outFileName="phrase-fst.txt") 
        elif type=="Smart":
            trainngram.trainTmSmart(table,outFileName="phrase-fst.txt") 

    def trainTmPhraseSymbolic(self,topK=1,renormalize=False):
        table={}
        highlyConfidentPhrases=0
        for phrase_en in self.phraseProbs:
            eKey=tuple([self.reverse_wids_en[x] for x in phrase_en])
            table[eKey]={}
            topValues=[(phrase_de,val) for phrase_de,val in self.phraseProbs[phrase_en].items()]
            topValues.sort(key = lambda x:-x[1])
           
            
            topValues=topValues[:topK]
            
            if renormalize:
                Z=sum([x[1] for x in topValues])
                topValues=[(phrase_de,val/Z) for phrase_de,val in topValues]
            
            highlyConfidentPhrases+=1
            
            for phrase_de,val in topValues:
                fKey=tuple([self.reverse_wids_de[x] for x in phrase_de])
                table[eKey][fKey]=val
        
        print "Number of Highly Confident Phrases",highlyConfidentPhrases

        table2={}
        for word_en in self.probs:
            eKey=self.reverse_wids_en[word_en]
            table2[eKey]={}
            topValues=[(word_de,val) for word_de,val in self.probs[word_en].items()]
            topValues.sort(key = lambda x:-x[1])
            for word_de,val in topValues[:10]:
                fKey=self.reverse_wids_de[word_de]
                table2[eKey][fKey]=val
 
        
        import trainngram
        trainngram.trainTmPhrase2(table,table2,outFileName="phrase-fst.txt",useTable2=True) 
       
 

if __name__=="__main__":
    trainAgain=False
    config=Config(EM_EPOCHS=10)
    predictor=Model1(config=config)

    if trainAgain:
        predictor.train()
        predictor.writeProbsToFile()
        predictor.saveModel()
    else:
        predictor.loadModel()

    predictor.readValidationData()
    predictor.trainPhrases(trainOn="valid",enPhraseLimit=4,frPhraseLimit=6,probabilistic=False)
    predictor.trainTmPhraseSymbolic(topK=10)
    #predictor.trainTmSymbolic(type="Smart")
    #predictor.convertAndWriteSentences("test")
    #predictor.convertAndWriteSentences("blind")

    #predictor.trainNGramSymbolic()
    #predictor.trainTmSymbolic()

    #predictor.trainPhrases()
