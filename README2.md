# New features
## Add new dictionary class for phoneme dictionary
## Add new model and example fastspeech
### Example ```fastspeech```
```
for SPLIT in train valid test; do \
        python -m examples.fastspeech.multiprocessing_bpe_encoder \
        --vocab-bpe experiments/phoneme_bpe/vocab.bpe \
        --inputs experiments/news-2017-19.en/news.${SPLIT}.txt \
        --outputs experiments/news-2017-19.en/news.${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done
```
## Add joint training with subword phoneme
- adding ```load_two_dictionary``` method to ```fairseq_task.py```
    
### Phoneme bpe files and Data
- ```./experiment/phoneme_bpe/vocab.bpe```, this file is for bpe pairs creation
-  ```./experiment/phoneme_bpe/dict.txt```, this file is for bpe dictionary creation
- ```experiments/news-2017-19.en/news.train.bpe```, this file is for saving raw data, ```{bpe-sequence} $ {phoneme sequence}```
    
### new file for prepare phoneme bpe
    
### preprocessing file
 - adding new arg ```--two-inputs``` to ```preprocess.py``` file
 
 
 
 
 python preprocess.py \
    --only-source \
    --srcdict experiments/phoneme/dict.txt \
    --tgtdict experiments/phoneme_bpe/dict.txt \
    --trainpref experiments/news-2017-19.en/news.train.test.bpe \
    --validpref experiments/news-2017-19.en/news.valid.test.bpe \
    --testpref experiments/news-2017-19.en/news.test.test.bpe \
    --destdir experiments/data-bin/news-2017-19.en.bpe \
    --dataset-impl dict \
    --two-inputs \
    --workers 2