set -e

for idx in 0 1 2 3 4 5 6 7 8 9; do
  seed=$((idx+1))

  PYTHONPATH=. python3 fine-tuning/run_classifier.py \
    --pretrained_model_path /mnt/ntfms/ET-BERT/models/lm/pre-trained_lm.bin-500000 \
    --vocab_path models/encryptd_vocab.txt \
    --train_path /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/vpn-service-flow/train_dataset.tsv \
    --dev_path   /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/vpn-service-flow/valid_dataset.tsv \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/vpn-service-flow/test_dataset.tsv \
    --output_model_path /mnt/ntfms/ET-BERT/models/lm/ft-vpn-serv/finetuned-vpn-serv-flow${idx}.bin \
    --epochs_num 10 --batch_size 32 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    --seq_length 128 --learning_rate 6e-5 \
    --seed ${seed} \
    2>&1 | tee /mnt/ntfms/ET-BERT/models/lm/ft-vpn-serv/finetune_lm_vpn-serv-flow${idx}.log

  sleep 60

  PYTHONPATH=. python3 fine-tuning/run_classifier.py \
    --pretrained_model_path /mnt/ntfms/ET-BERT/models/lm/pre-trained_lm.bin-500000 \
    --vocab_path models/encryptd_vocab.txt \
    --train_path /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/ustc-app-flow/train_dataset.tsv \
    --dev_path   /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/ustc-app-flow/valid_dataset.tsv \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/ustc-app-flow/test_dataset.tsv \
    --output_model_path /mnt/ntfms/ET-BERT/models/lm/ft-ustc-app/finetuned-ustc-app-flow${idx}.bin \
    --epochs_num 10 --batch_size 32 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    --seq_length 128 --learning_rate 6e-5 \
    --seed ${seed} \
    2>&1 | tee /mnt/ntfms/ET-BERT/models/lm/ft-ustc-app/finetune_lm_ustc-app-flow${idx}.log

  sleep 60

  PYTHONPATH=. python3 fine-tuning/run_classifier.py \
    --pretrained_model_path /mnt/ntfms/ET-BERT/models/lm/pre-trained_lm.bin-500000 \
    --vocab_path models/encryptd_vocab.txt \
    --train_path /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/tor-service-flow/train_dataset.tsv \
    --dev_path   /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/tor-service-flow/valid_dataset.tsv \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/tor-service-flow/test_dataset.tsv \
    --output_model_path /mnt/ntfms/ET-BERT/models/lm/ft-tor/finetuned-tor-flow${idx}.bin \
    --epochs_num 10 --batch_size 32 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    --seq_length 128 --learning_rate 6e-5 \
    --seed ${seed} \
    2>&1 | tee /mnt/ntfms/ET-BERT/models/lm/ft-tor/finetune_lm_tor-flow${idx}.log

  sleep 60

  PYTHONPATH=. python3 fine-tuning/run_classifier.py \
    --pretrained_model_path /mnt/ntfms/ET-BERT/models/lm/pre-trained_lm.bin-500000 \
    --vocab_path models/encryptd_vocab.txt \
    --train_path /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/ios-app-flow/train_dataset.tsv \
    --dev_path   /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/ios-app-flow/valid_dataset.tsv \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/ios-app-flow/test_dataset.tsv \
    --output_model_path /mnt/ntfms/ET-BERT/models/lm/ft-ios/finetuned-ios-flow${idx}.bin \
    --epochs_num 10 --batch_size 32 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    --seq_length 128 --learning_rate 6e-5 \
    --seed ${seed} \
    2>&1 | tee /mnt/ntfms/ET-BERT/models/lm/ft-ios/finetune_lm_ios-flow${idx}.log

  sleep 60

  PYTHONPATH=. python3 fine-tuning/run_classifier.py \
    --pretrained_model_path /mnt/ntfms/ET-BERT/models/lm/pre-trained_lm.bin-500000 \
    --vocab_path models/encryptd_vocab.txt \
    --train_path /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/csnet-tls-flow/train_dataset.tsv \
    --dev_path   /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/csnet-tls-flow/valid_dataset.tsv \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/csnet-tls-flow/test_dataset.tsv \
    --output_model_path /mnt/ntfms/ET-BERT/models/lm/ft-cstnet/finetuned-cstnet-flow${idx}.bin \
    --epochs_num 10 --batch_size 32 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    --seq_length 128 --learning_rate 6e-5 \
    --seed ${seed} \
    2>&1 | tee /mnt/ntfms/ET-BERT/models/lm/ft-cstnet/finetune_lm_cstnet-flow${idx}.log

  sleep 60

  PYTHONPATH=. python3 fine-tuning/run_classifier.py \
    --pretrained_model_path /mnt/ntfms/ET-BERT/models/lm/pre-trained_lm.bin-500000 \
    --vocab_path models/encryptd_vocab.txt \
    --train_path /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/andro-app-flow/train_dataset.tsv \
    --dev_path   /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/andro-app-flow/valid_dataset.tsv \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/andro-app-flow/test_dataset.tsv \
    --output_model_path /mnt/ntfms/ET-BERT/models/lm/ft-andro/finetuned-andro-flow${idx}.bin \
    --epochs_num 10 --batch_size 32 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    --seq_length 128 --learning_rate 6e-5 \
    --seed ${seed} \
    2>&1 | tee /mnt/ntfms/ET-BERT/models/lm/ft-andro/finetune_lm_andro-flow${idx}.log

  sleep 60

  PYTHONPATH=./fine-tuning:$PYTHONPATH python3 inference/run_classifier_infer.py \
    --load_model_path /mnt/ntfms/ET-BERT/models/lm/ft-vpn-serv/finetuned-vpn-serv-flow${idx}.bin \
    --vocab_path models/encryptd_vocab.txt \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/vpn-service-flow/nolabel_test_dataset.tsv \
    --prediction_path /mnt/ntfms/ET-BERT/models/lm/ft-vpn-serv/prediction${idx}-vpn-serv.tsv \
    --labels_num 6 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    2>&1 | tee /mnt/ntfms/ET-BERT/models/lm/ft-vpn-serv/results-inference${idx}.log

  sleep 60

  PYTHONPATH=./fine-tuning:$PYTHONPATH python3 inference/run_classifier_infer.py \
    --load_model_path /mnt/ntfms/ET-BERT/models/lm/ft-ustc-app/finetuned-ustc-app-flow${idx}.bin \
    --vocab_path models/encryptd_vocab.txt \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/ustc-app-flow/nolabel_test_dataset.tsv \
    --prediction_path /mnt/ntfms/ET-BERT/models/lm/ft-ustc-app/prediction${idx}-ustc-app.tsv \
    --labels_num 20 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    2>&1 | tee /mnt/ntfms/ET-BERT/models/lm/ft-ustc-app/results-inference${idx}.log

  sleep 60

  PYTHONPATH=./fine-tuning:$PYTHONPATH python3 inference/run_classifier_infer.py \
    --load_model_path /mnt/ntfms/ET-BERT/models/lm/ft-tor/finetuned-tor-flow${idx}.bin \
    --vocab_path models/encryptd_vocab.txt \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/tor-service-flow/nolabel_test_dataset.tsv \
    --prediction_path /mnt/ntfms/ET-BERT/models/lm/ft-tor/prediction${idx}-tor.tsv \
    --labels_num 7 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    2>&1 | tee /mnt/ntfms/ET-BERT/models/lm/ft-tor/results-inference${idx}.log

  sleep 60

  PYTHONPATH=./fine-tuning:$PYTHONPATH python3 inference/run_classifier_infer.py \
    --load_model_path /mnt/ntfms/ET-BERT/models/lm/ft-ios/finetuned-ios-flow${idx}.bin \
    --vocab_path models/encryptd_vocab.txt \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/ios-app-flow/nolabel_test_dataset.tsv \
    --prediction_path /mnt/ntfms/ET-BERT/models/lm/ft-ios/prediction${idx}-ios.tsv \
    --labels_num 196 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    2>&1 | tee /mnt/ntfms/ET-BERT/models/lm/ft-ios/results-inference${idx}.log

  sleep 60

  PYTHONPATH=./fine-tuning:$PYTHONPATH python3 inference/run_classifier_infer.py \
    --load_model_path /mnt/ntfms/ET-BERT/models/lm/ft-cstnet/finetuned-cstnet-flow${idx}.bin \
    --vocab_path models/encryptd_vocab.txt \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/csnet-tls-flow/nolabel_test_dataset.tsv \
    --prediction_path /mnt/ntfms/ET-BERT/models/lm/ft-cstnet/prediction${idx}-cstnet.tsv \
    --labels_num 120 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    2>&1 | tee /mnt/ntfms/ET-BERT/models/lm/ft-cstnet/results-inference${idx}.log

  sleep 60

  PYTHONPATH=./fine-tuning:$PYTHONPATH python3 inference/run_classifier_infer.py \
    --load_model_path /mnt/ntfms/ET-BERT/models/lm/ft-andro/finetuned-andro-flow${idx}.bin \
    --vocab_path models/encryptd_vocab.txt \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/andro-app-flow/nolabel_test_dataset.tsv \
    --prediction_path /mnt/ntfms/ET-BERT/models/lm/ft-andro/prediction${idx}-andro.tsv \
    --labels_num 212 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    2>&1 | tee /mnt/ntfms/ET-BERT/models/lm/ft-andro/results-inference${idx}.log

  sleep 60
done
