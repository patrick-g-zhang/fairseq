# New features
## Add new dictionary class for phoneme dictionary
## Add new models fastspeech
## Add joint training with subword phoneme
    - adding ```load_two_dictionary``` method to ```fairseq_task.py```
    - adding new arg ```--two-inputs``` to ```preprocess.py``` file

    ### phoneme bpe files
    1. ./experiment/phoneme_bpe/vocab.bpe, this file is for bpe pairs
    2. ./experiment/phoneme_bpe/dict.txt, this file is for bpe dict creation

    ### new file for prepare phoneme bpe

    ### preprocessing file
    