# ALL scripts in this file come from Autoformer
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

model_name=Transformer
decay_type="powerLaw"
decay_scale=0.5
seq_len=256
pred_len=96
itr=0
random_seed=2021

root_path_name=./datasets/
data_path_name=weather/weather.csv
model_id_name=Weather
data_name=custom

python3 -u run_longExp.py \
    --attn_decay_type ${decay_type} \
    --attn_decay_scale ${decay_scale} \
    --random_seed ${random_seed} \
    --is_training 1 \
    --root_path ${root_path_name} \
    --data_path ${data_path_name} \
    --model_id ${model_id_name} \
    --model ${model_name} \
    --data ${data_name} \
    --features M \
    --seq_len ${seq_len} \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr ${itr} \
    "$@" >> logs/LongForecasting/${model_name}'_'${model_id_name}'_'${seq_len}'_'${pred_len}.log