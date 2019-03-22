python run_NN.py --config configs/cnn_14.json --out model_v1.13 --debug

sleep 60s
gcloud compute instances stop instance-gpu --zone asia-east1-c
