## Train Phoneme Bert with Fastspeech Text Encoder(ADD PROSODY PREDICTOR)

### Add new example ```fastspeech``` for preparing data
- The new example convert phoneme string to bpe code.
- we remove the UNK and EOS of phoneme string and add <s> and </s> as start and end of the bpe sequences
- the ```vocab.bpe``` is bpe pair for bpe creation, the dictionary size is 30k
- convert phoneme sequence to bpe sequence
- 27th March, add ```no-word-sep``` as argument which indicates the word splitting not using
- convert phoneme sequence to bpe sequence dictionary size is 10k
- add prosody predictor part ```multiprocessing_bpe_prosody_encoder.py```

```
for SPLIT in train valid test; do \
        python -m examples.fastspeech.multiprocessing_bpe_prosody_encoder \
        --vocab-bpe experiments/phoneme_bpe/vocab.10k.bpe \
        --inputs experiments/librispeech-prosody/${SPLIT} \
        --outputs experiments/librispeech-prosody/${SPLIT}_processed \
        --keep-empty \
        --workers 40; \
done
```

- add new argument --phoneme-prosody 
- new data dir librispeech-prosody-interplote
```
for SPLIT in train valid test; do \
        python -m examples.fastspeech.multiprocessing_bpe_prosody_encoder \
        --vocab-bpe experiments/phoneme_bpe/vocab.10k.bpe \
        --inputs experiments/librispeech-prosody-interplote/${SPLIT} \
        --outputs experiments/librispeech-prosody-interplote/${SPLIT}_processed \
        --keep-empty \
        --phoneme-prosody \
        --workers 40; \
done
```
    
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

- phoneme level prosody
 ```
    SRCDICT=experiments/phoneme/dict.txt # phoneme dictionary
    TGTDICT=experiments/phoneme_bpe/bpe.10k.dict.txt # bpe dictionary
    TRAINPREF=experiments/librispeech-prosody-interplote/train_processed
    VALIDPREF=experiments/librispeech-prosody-interplote/valid_processed
    TESTPREF=experiments/librispeech-prosody-interplote/test_processed
    DESTDIR=experiments/data-bin/librispeech-prosody-interplote # output
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
    --workers 30
 ```



### Training Command
 - adding new argument ```--two-inputs``` to ```train.py``` file. The input needs both bpe and phoneme
 - adding new argument ```--prosody-predict``` to ```train.py``` file. we will also predict prosody.
 - adding ```tensorboard-logdir```
 - adding ```num-spk``` 
 - adding ```prosody-loss-coeff```
#### General configs
```
    ARCH=fastspeech # 我把韵律预测与归到这里了
    SAVE_DIR=/blob/xuta/speech/tts/t-guzhang/fairseq/checkpoints/${ARCH}-Test
    DATA_DIR=experiments/data-bin/librispeech-prosody
    LOG_DIR="logs/fastspeech-Test"
    TOTAL_UPDATES=225000    # Total number of training steps 
    WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
    PEAK_LR=0.0005          # Peak learning rate, adjust as needed
    TOKENS_PER_SAMPLE=512   # Max sequence length 
    MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
    MAX_SENTENCES=16        # Number of sequences per batch (batch size)
    UPDATE_FREQ=16          # Increase the batch size 16x  
    NUM_SPK=2485 # number of speaker for librispeech 1000
```


 ```
 python /blob/xuta/speech/tts/t-guzhang/fairseq/train.py $DATA_DIR \
    --task masked_lm --criterion masked_lm --save-dir $SAVE_DIR\
    --arch $ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ  \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 --dataset-impl dict --two-inputs --no-pad-prepend-token --prosody-predict --num-spk $NUM_SPK --mask-whole-words
 ```




### Warm start from a pretrained MLM model
- add ```load_pretrained_checkpoint ``` to ```checkpoint_utils.py```
- 确保存在 ```checkpointpretrained.pt ``` 的时候并且没有其他的checkpoint的时候进行warm start
 

### Adding some parameters for prosody predictor coefficient
```
    ARCH=fastspeech # 我把韵律预测与归到这里了
    SAVE_DIR=/blob/xuta/speech/tts/t-guzhang/fairseq/checkpoints/${ARCH}-Test
    DATA_DIR=experiments/data-bin/librispeech-prosody
    LOG_DIR="logs/fastspeech-Test"
    TOTAL_UPDATES=225000    # Total number of training steps 
    WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
    PEAK_LR=0.0005          # Peak learning rate, adjust as needed
    TOKENS_PER_SAMPLE=512   # Max sequence length 
    MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
    MAX_SENTENCES=16        # Number of sequences per batch (batch size)
    UPDATE_FREQ=16          # Increase the batch size 16x  
    NUM_SPK=2485 # number of speaker for librispeech 1000
    PCOEFF=1000
```

 ```
 python -m pdb /blob/xuta/speech/tts/t-guzhang/fairseq/train.py $DATA_DIR \
    --task masked_lm --criterion masked_lm --save-dir $SAVE_DIR\
    --arch $ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ  \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 --dataset-impl dict --two-inputs --no-pad-prepend-token --prosody-predict --num-spk $NUM_SPK --mask-whole-words --prosody-loss-coeff 1000
 ```

### Phoneme level prosody predictor
```
    ARCH=fastspeech # 我把韵律预测与归到这里了
    SAVE_DIR=/blob/xuta/speech/tts/t-guzhang/fairseq/checkpoints/${ARCH}-Test
    DATA_DIR=experiments/data-bin/librispeech-prosody-interplote
    LOG_DIR="logs/fastspeech-Test"
    TOTAL_UPDATES=125000    # Total number of training steps 
    WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
    PEAK_LR=0.0005          # Peak learning rate, adjust as needed
    TOKENS_PER_SAMPLE=512   # Max sequence length 
    MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
    MAX_SENTENCES=16        # Number of sequences per batch (batch size)
    UPDATE_FREQ=16          # Increase the batch size 16x  
    NUM_SPK=2485 # number of speaker for librispeech 1000
```

 ```
 python -m pdb /blob/xuta/speech/tts/t-guzhang/fairseq/train.py $DATA_DIR \
    --task masked_lm --criterion masked_lm --save-dir $SAVE_DIR\
    --arch $ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ  \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 --dataset-impl dict --two-inputs --no-pad-prepend-token --prosody-predict --num-spk $NUM_SPK --mask-whole-words --phoneme-prosody
 ```
