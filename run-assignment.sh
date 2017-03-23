#python Model1.py

python symbols.py 2 < phrase-fst.txt > phrase-fst.isym
python symbols.py 2 < ngram-fst.txt  > ngram-fst.isym
fstcompile --isymbols=ngram-fst.isym --osymbols=ngram-fst.isym ngram-fst.txt | fstarcsort > ngram-fst.fst
fstcompile --isymbols=phrase-fst.isym --osymbols=ngram-fst.isym phrase-fst.txt | fstarcsort > phrase-fst.fst

python decode.py phrase-fst.fst ngram-fst.fst phrase-fst.isym ngram-fst.isym blind.convert blind.convert.en
#python decode.py phrase-fst.fst ngram-fst.fst phrase-fst.isym ngram-fst.isym blind.convert blind.convert.en
