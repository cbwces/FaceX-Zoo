DATE=`date +%y%m%d`
mkdir $DATE
python train_amp.py \
    --data_root '/home/cbw233/datasets/face_recognition/train/msra/msra_crop' \
    --train_file '/home/cbw233/datasets/face_recognition/train/msra/msra_data.txt' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../backbone_conf.yaml' \
    --head_type 'MagFace' \
    --head_conf_file '../head_conf.yaml' \
    --lr 0.001 \
    --out_dir "$DATE" \
    --epoches 18 \
    --step '10, 13, 16' \
    --print_freq 400 \
    --batch_size 512 \
    --virtual_batch \
    --momentum 0.9 \
    --pretrain_model './pretrained/Epoch_7.pt' \
    --log_dir "$DATE"
