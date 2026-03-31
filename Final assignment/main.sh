wandb login $WANDB_API_KEY

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --encoder-lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "Unet+ResNet" \