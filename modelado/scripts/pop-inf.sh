set -e

PRETRAIN=/mnt/ntfms/ET-BERT/models/pop/pre-trained_pop.bin-500000
VOCAB=models/encryptd_vocab.txt
ROOT_OUT=/mnt/ntfms/ET-BERT/models/pop

for idx in 0 1 2 3 4 5 6 7 8 9; do
  seed=$((idx+1))

  PYTHONPATH=. python3 fine-tuning/run_classifier.py \
    --pretrained_model_path "$PRETRAIN" \
    --vocab_path $VOCAB \
    --train_path /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/vpn-service-flow-pop/train_dataset.tsv \
    --dev_path   /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/vpn-service-flow-pop/valid_dataset.tsv \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/vpn-service-flow-pop/test_dataset.tsv \
    --output_model_path $ROOT_OUT/ft-vpn-serv/finetuned-vpn-serv-flow${idx}.bin \
    --epochs_num 10 --batch_size 32 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    --seq_length 128 --learning_rate 6e-5 \
    --seed ${seed} \
    2>&1 | tee $ROOT_OUT/ft-vpn-serv/finetune_pop_vpn-serv-flow${idx}.log
  sleep 15

  PYTHONPATH=. python3 fine-tuning/run_classifier.py \
    --pretrained_model_path "$PRETRAIN" \
    --vocab_path $VOCAB \
    --train_path /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/vpn-app-flow-pop/train_dataset.tsv \
    --dev_path   /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/vpn-app-flow-pop/valid_dataset.tsv \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/vpn-app-flow-pop/test_dataset.tsv \
    --output_model_path $ROOT_OUT/ft-vpn-app/finetuned-vpn-app-flow${idx}.bin \
    --epochs_num 10 --batch_size 32 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    --seq_length 128 --learning_rate 6e-5 \
    --seed ${seed} \
    2>&1 | tee $ROOT_OUT/ft-vpn-app/finetune_pop_vpn-app-flow${idx}.log
  sleep 15

  PYTHONPATH=. python3 fine-tuning/run_classifier.py \
    --pretrained_model_path "$PRETRAIN" \
    --vocab_path $VOCAB \
    --train_path /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/ustc-app-flow-pop/train_dataset.tsv \
    --dev_path   /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/ustc-app-flow-pop/valid_dataset.tsv \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/ustc-app-flow-pop/test_dataset.tsv \
    --output_model_path $ROOT_OUT/ft-ustc-app/finetuned-ustc-app-flow${idx}.bin \
    --epochs_num 10 --batch_size 32 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    --seq_length 128 --learning_rate 6e-5 \
    --seed ${seed} \
    2>&1 | tee $ROOT_OUT/ft-ustc-app/finetune_pop_ustc-app-flow${idx}.log
  sleep 15

  PYTHONPATH=. python3 fine-tuning/run_classifier.py \
    --pretrained_model_path "$PRETRAIN" \
    --vocab_path $VOCAB \
    --train_path /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/tor-service-flow-pop/train_dataset.tsv \
    --dev_path   /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/tor-service-flow-pop/valid_dataset.tsv \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/tor-service-flow-pop/test_dataset.tsv \
    --output_model_path $ROOT_OUT/ft-tor/finetuned-tor-flow${idx}.bin \
    --epochs_num 10 --batch_size 32 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    --seq_length 128 --learning_rate 6e-5 \
    --seed ${seed} \
    2>&1 | tee $ROOT_OUT/ft-tor/finetune_pop_tor-flow${idx}.log
  sleep 15

  PYTHONPATH=. python3 fine-tuning/run_classifier.py \
    --pretrained_model_path "$PRETRAIN" \
    --vocab_path $VOCAB \
    --train_path /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/ios-app-flow-pop/train_dataset.tsv \
    --dev_path   /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/ios-app-flow-pop/valid_dataset.tsv \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/ios-app-flow-pop/test_dataset.tsv \
    --output_model_path $ROOT_OUT/ft-ios/finetuned-ios-flow${idx}.bin \
    --epochs_num 10 --batch_size 32 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    --seq_length 128 --learning_rate 6e-5 \
    --seed ${seed} \
    2>&1 | tee $ROOT_OUT/ft-ios/finetune_pop_ios-flow${idx}.log
  sleep 15

  PYTHONPATH=. python3 fine-tuning/run_classifier.py \
    --pretrained_model_path "$PRETRAIN" \
    --vocab_path $VOCAB \
    --train_path /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/andro-app-flow-pop/train_dataset.tsv \
    --dev_path   /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/andro-app-flow-pop/valid_dataset.tsv \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/andro-app-flow-pop/test_dataset.tsv \
    --output_model_path $ROOT_OUT/ft-andro/finetuned-andro-flow${idx}.bin \
    --epochs_num 10 --batch_size 32 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    --seq_length 128 --learning_rate 6e-5 \
    --seed ${seed} \
    2>&1 | tee $ROOT_OUT/ft-andro/finetune_pop_andro-flow${idx}.log
  sleep 15

  PYTHONPATH=./fine-tuning:$PYTHONPATH python3 inference/run_classifier_infer.py \
    --load_model_path $ROOT_OUT/ft-vpn-serv/finetuned-vpn-serv-flow${idx}.bin \
    --vocab_path $VOCAB \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/vpn-service-flow-pop/nolabel_test_dataset.tsv \
    --prediction_path $ROOT_OUT/ft-vpn-serv/prediction${idx}-vpn-serv.tsv \
    --labels_num 6 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    2>&1 | tee $ROOT_OUT/ft-vpn-serv/results-inference${idx}.log
  sleep 15

  PYTHONPATH=./fine-tuning:$PYTHONPATH python3 inference/run_classifier_infer.py \
    --load_model_path $ROOT_OUT/ft-vpn-app/finetuned-vpn-app-flow${idx}.bin \
    --vocab_path $VOCAB \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/vpn-app-flow-pop/nolabel_test_dataset.tsv \
    --prediction_path $ROOT_OUT/ft-vpn-app/prediction${idx}-vpn-app.tsv \
    --labels_num 14 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    2>&1 | tee $ROOT_OUT/ft-vpn-app/results-inference${idx}.log
  sleep 15

  PYTHONPATH=./fine-tuning:$PYTHONPATH python3 inference/run_classifier_infer.py \
    --load_model_path $ROOT_OUT/ft-ustc-app/finetuned-ustc-app-flow${idx}.bin \
    --vocab_path $VOCAB \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/ustc-app-flow-pop/nolabel_test_dataset.tsv \
    --prediction_path $ROOT_OUT/ft-ustc-app/prediction${idx}-ustc-app.tsv \
    --labels_num 20 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    2>&1 | tee $ROOT_OUT/ft-ustc-app/results-inference${idx}.log
  sleep 15

  PYTHONPATH=./fine-tuning:$PYTHONPATH python3 inference/run_classifier_infer.py \
    --load_model_path $ROOT_OUT/ft-tor/finetuned-tor-flow${idx}.bin \
    --vocab_path $VOCAB \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/tor-service-flow-pop/nolabel_test_dataset.tsv \
    --prediction_path $ROOT_OUT/ft-tor/prediction${idx}-tor.tsv \
    --labels_num 7 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    2>&1 | tee $ROOT_OUT/ft-tor/results-inference${idx}.log
  sleep 15

  PYTHONPATH=./fine-tuning:$PYTHONPATH python3 inference/run_classifier_infer.py \
    --load_model_path $ROOT_OUT/ft-ios/finetuned-ios-flow${idx}.bin \
    --vocab_path $VOCAB \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/ios-app-flow-pop/nolabel_test_dataset.tsv \
    --prediction_path $ROOT_OUT/ft-ios/prediction${idx}-ios.tsv \
    --labels_num 196 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    2>&1 | tee $ROOT_OUT/ft-ios/results-inference${idx}.log
  sleep 15

  PYTHONPATH=./fine-tuning:$PYTHONPATH python3 inference/run_classifier_infer.py \
    --load_model_path $ROOT_OUT/ft-andro/finetuned-andro-flow${idx}.bin \
    --vocab_path $VOCAB \
    --test_path  /mnt/ntfms/NetBench/datasets-gdrive/Flow-Level/andro-app-flow-pop/nolabel_test_dataset.tsv \
    --prediction_path $ROOT_OUT/ft-andro/prediction${idx}-andro.tsv \
    --labels_num 212 --embedding word_pos_seg \
    --encoder transformer --mask fully_visible \
    2>&1 | tee $ROOT_OUT/ft-andro/results-inference${idx}.log
  sleep 15
done
