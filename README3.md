## Train Phoneme Bert with Fastspeech Text Encoder
### Add prosody predictor 

### Add new model and example ```fastspeech```
- move code files from fastspeech repository to fairseq ```fairseq/models/fastspeech```
- change the value of argument ```ARCH``` to ```fastspeech```

```
python $dist_config /blob/xuta/speech/tts/t-guzhang/fairseq/train.py --fp16 $DATA_DIR \
        --save-dir $SAVE_DIR \
    --task masked_lm --criterion masked_lm  \
    --arch $ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --skip-invalid-size-inputs-valid-test \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --continuous-mask $CON_MASK --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 50 --ddp-backend=no_c10d \
        --distributed-backend 'nccl' --distributed-no-spawn \
        --no-pad-prepend-token --phoneme-dict \
        2>&1 | tee -a ${SAVE_DIR}/train.log
```
### Add new dictionary class for phoneme dictionary
- 27th March: add ‘nosep’ to phoneme dictionary indicates there is no word split, 'dict.txt' to 'dict_noseq.txt' 

## BPE and phoneme bert joint training


### Add new example ```fastspeech``` for preparing data
- The new example convert phoneme string to bpe code.
- we remove the UNK and EOS of phoneme string and add <s> and </s> as start and end of the bpe sequences
- the ```vocab.bpe``` is bpe pair for bpe creation, the dictionary size is 30k
- convert phoneme sequence to bpe sequence
- 27th March, add ```no-word-sep``` as argument which indicates the word splitting not using
- convert phoneme sequence to bpe sequence dictionary size is 10k
- add prosody predictor part
```
for SPLIT in train valid test; do \
        python -m examples.fastspeech.multiprocessing_bpe_prosody_encoder \
        --vocab-bpe experiments/phoneme_bpe/vocab.10k.bpe \
        --inputs experiments/librispeech-prosody/${SPLIT} \
        --outputs experiments/librispeech-prosody/${SPLIT}_processed \
        --keep-empty \
        --workers 20; \
done
```

### Add joint training with subword phoneme
- adding ```load_two_dictionary``` method to ```fairseq_task.py```
    
### Phoneme bpe files and Data
- ```./experiment/phoneme_bpe/vocab.bpe```, this file is for bpe pairs creation
-  ```./experiment/phoneme_bpe/dict.txt```, this file is for bpe dictionary
-  ```./experiment/phoneme_bpe/bpe.30k.dict.txt```, this file is for bpe dictionary with corrected bpe method, avoid unk problem !!!!
-   creation
- ```experiments/news-2017-19.en/news.train.bpe```, this file is for saving raw data, ```{bpe-sequence} $ {phoneme sequence}```
    



### Preprocessing Command- preprocess the raw bpe data to binary file
 - adding new argument ```--two-inputs``` to ```preprocess.py``` file. 
 - create a new index dataset named ```DictIndexedDataset```, since we need to store phoneme sequence, sub-word sequence, and phoneme2sub-word. The three vectors are stored in dictionary format.
 - changing argument ```--dataset-impl``` to ```dict```
 - adding new argument ```--indexed-dataset``` to indicate the input is indexed dataset instead of text 

 
  - preprocessing full dataset with vocab size 10k
 ```
    SRCDICT=experiments/phoneme/dict.txt # phoneme dictionary
    TGTDICT=experiments/phoneme_bpe/bpe.10k.dict.txt # bpe dictionary
    TRAINPREF=experiments/librispeech-prosody/train_processed
    VALIDPREF=experiments/librispeech-prosody/valid_processed
    TESTPREF=experiments/librispeech-prosody/test_processed
    DESTDIR=experiments/data-bin/librispeech-prosody # output
 ```
    
 - run command
 ```
      python preprocess.py \
    --only-source \
    --srcdict ${SRCDICT} \
    --tgtdict ${TGTDICT} \
    --trainpref ${TRAINPREF} \
    --validpref  ${VALIDPREF} \
    --testpref  ${TESTPREF}\
    --destdir  ${DESTDIR}\
    --dataset-impl dict \
    --two-inputs \
    --indexed-dataset\
    --workers 3
 ```

### Binary data to training dataset
- Class ```BPEMaskTokensDataset``` , This dataset implementation is for dictionary masked input.
- adding ```--mask-whole-words``` argument


### Training Command
 - adding new argument ```--two-inputs``` to ```train.py``` file. The input needs both bpe and phoneme
 - adding new argument ```--prosody-predict``` to ```train.py``` file. we will also predict prosody.
 - adding ```tensorboard-logdir```
 
#### General configs
```
    ARCH=fastspeech # 我把韵律预测与归到这里了
    SAVE_DIR=/blob/xuta/speech/tts/t-guzhang/fairseq/checkpoints/${ARCH}-Test
    DATA_DIR=/blob/xuta/speech/tts/t-guzhang/fairseq/experiments/data-bin/news.cn.bpe.10k.full
    LOG_DIR="logs/fastspeech-Test"
```




#### Full data, second version dictionary, 30k size with full data, 
```
SAVE_DIR=/blob/xuta/speech/tts/t-guzhang/fairseq/checkpoints/${ARCH}-BPE-12W-Steps-FP16-wu3
DATA_DIR=/blob/xuta/speech/tts/t-guzhang/fairseq/experiments/data-bin/news-2017-19.en.bpe.30k.full
LOG_DIR=/blob/xuta/speech/tts/t-guzhang/fairseq/logs/${ARCH}-BPE-12W-Steps-FP16-wu3
```

 ```
 python  /blob/xuta/speech/tts/t-guzhang/fairseq/train.py $DATA_DIR \
    --task masked_lm --criterion masked_lm --save-dir $SAVE_DIR\
    --arch $ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ  \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 --dataset-impl dict --two-inputs --no-pad-prepend-token --tensorboard-logdir=$
 ```
 
  ```
 python /blob/xuta/speech/tts/t-guzhang/fairseq/train.py $DATA_DIR \
    --task masked_lm --criterion masked_lm --save-dir $SAVE_DIR\
    --arch $ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ  \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 --dataset-impl dict --two-inputs --no-pad-prepend-token
 ```
 
 
