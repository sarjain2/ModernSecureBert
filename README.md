# SECUREBERT 2.0: AN ENHANCED ENCODER-BASED DOMAIN-SPECIFIC LANGUAGE MODEL

Weights saved in Google drive of each task -> planning to deploy to HF after approval

https://drive.google.com/drive/folders/1RQkuU-kLtAb1VOuvqNvANv99OR6KFwNC?usp=share_link

Filepaths in lightning ai:

Vulnerability Classification: secure_modern_bert/sentiment_classif/checkpoint_epoch_10.pth

Cross Encoder: secure_modern_bert/embedding_sim_weights/checkpoint_epoch_9.pth

Dual Encoder: secure_modern_bert/modernbert_dual_encoder_mnr/model.safetensors

MLM (base model): secure_modern_bert/final_base_modernsecurebert_pths/checkpoint_epoch_20.pth

Ner: secure_modern_bert/final_ner_original/without_test/checkpoint_epoch_20.pth (Without test of augmented set in)

All possible train data NER: secure_modern_bert/final_ner_original/take_2/checkpoint_epoch_20.pth

dataset.py - Global Dataset for DataLoading script

cont_load.py and primus_load.py - load all datasets from HuggingFaceHub to local disk

All these below scripts got if __name__ == "__main__"

train.py - Train MLM (SecureBert 2.0) starting from ModernBert

code_mlm_eval.py - Code MLM evaluation for SecureBert 2.0

SecureBert2_mlm_eval.py - Evaluation Verbs.csv and Object_pred.csv for SecureBert 2.0

SecureBert_mlm_eval.py - Evaluation Verbs.csv and Object_pred.csv for SecureBert (Original)

run_modernbert.py - Single Instance Inference for Secure Bert 2.0 MLM

BiEncoder_eval.py - Evaluation for BiEncoder Downstream Task

BiEncoder_train.py - Training BiEncoder model

CrossEncoder_eval.py - Evaluation for Cross Encoder Downstream Task

CrossEncoder_infer.py - Inference for Cross Encoder Downstream Task

CrossEncoder_train.py - Training for Cross Encoder Downstream Task

NER_eval.py - Evaluation for NER Downstream Task

NER_infer.py - Single instance inference for NER Downstream Task

NER_train.py - Training for NER Downstream Task

CodeVuln_eval.py - Code Vulnernability Evaluation

CodeVuln_infer.py - Single Instance Inference for Code Vulnerability

vuln_sentiment_train.py - Train Code Vulnerability downstream task






