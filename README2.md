## Train Phoneme Bert with Fastspeech Text Encoder
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


## BPE and phoneme bert joint training


### Add new example ```fastspeech```
- The new example convert phoneme string to bpe code.
- the ```vocab.bpe``` is bpe pair for bpe creation
- convert phoneme sequence to bpe sequence
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

### Add joint training with subword phoneme
- adding ```load_two_dictionary``` method to ```fairseq_task.py```
    
### Phoneme bpe files and Data
- ```./experiment/phoneme_bpe/vocab.bpe```, this file is for bpe pairs creation
-  ```./experiment/phoneme_bpe/dict.txt```, this file is for bpe dictionary
-  ```./experiment/phoneme_bpe/dict.wp.txt```, this file is for bpe dictionary with single word included, avoid unk problem !!!!
-   creation
- ```experiments/news-2017-19.en/news.train.bpe```, this file is for saving raw data, ```{bpe-sequence} $ {phoneme sequence}```
    

### Class ```BPEMaskTokensDataset```
This dataset implementation is for dictionary masked input.

### Preprocessing Command
 - adding new argument ```--two-inputs``` to ```preprocess.py``` file. 
 - create a new index dataset named ```DictIndexedDataset```, since we need to store phoneme sequence, sub-word sequence, and phoneme2sub-word. The three vectors are stored in dictionary format.
 - changing argument ```--dataset-impl``` to ```dict```
 - preprocess short and test files


 ```
     python preprocess.py \
    --only-source \
    --srcdict experiments/phoneme/dict.txt \
    --tgtdict experiments/phoneme_bpe/bpe.wp.dict.txt \
    --trainpref experiments/news-2017-19.en/news.train.test.bpe \
    --validpref experiments/news-2017-19.en/news.valid.test.bpe \
    --testpref experiments/news-2017-19.en/news.test.test.bpe \
    --destdir experiments/data-bin/news-2017-19.en.bpe \
    --dataset-impl dict \
    --two-inputs \
    --workers 2
 ```
 - preprocessing full dataset
 ```
     python preprocess.py \
    --only-source \
    --srcdict experiments/phoneme/dict.txt \
    --tgtdict experiments/phoneme_bpe/bpe.wp.dict.txt \
    --trainpref experiments/news-2017-19.en/news.train.bpe \
    --validpref experiments/news-2017-19.en/news.valid.bpe \
    --testpref experiments/news-2017-19.en/news.test.bpe \
    --destdir experiments/data-bin/news-2017-19.en.bpe.wp.full \
    --dataset-impl dict \
    --two-inputs \
    --workers 64
 ```


### Training Command
 - adding new argument ```--two-inputs``` to ```train.py``` file. The input needs both bpe and phoneme
 - adding ```tensorboard-logdir```

#### Test with small data, 
```
SAVE_DIR=/blob/xuta/speech/tts/t-guzhang/fairseq/checkpoints/${ARCH}-Test
DATA_DIR=/blob/xuta/speech/tts/t-guzhang/fairseq/experiments/data-bin/news-2017-19.en.bpe
LOG_DIR="logs/fastspeech-Test"
```

#### Test with full data, 
```
SAVE_DIR=/blob/xuta/speech/tts/t-guzhang/fairseq/checkpoints/${ARCH}-BPE-12W-Steps-FP16-wu3
DATA_DIR=/blob/xuta/speech/tts/t-guzhang/fairseq/experiments/data-bin/news-2017-19.en.bpe.full
LOG_DIR=/blob/xuta/speech/tts/t-guzhang/fairseq/logs/${ARCH}-BPE-12W-Steps-FP16-wu3
```

 ```
 python $dist_config /blob/xuta/speech/tts/t-guzhang/fairseq/train.py --fp16  $DATA_DIR \
    --task masked_lm --criterion masked_lm --save-dir $SAVE_DIR\
    --arch $ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ  \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 --dataset-impl dict --two-inputs --no-pad-prepend-token --tensorboard-logdir=$
 ```
 
 
 
 
