echo "Original Model"
python train_cbm.py -dataset synthetic_2 -model_type joint -num_attributes 4 -num_classes 2 -seed 42 -epochs 50 --encoder_model inceptionv3 

echo "Regularized Model"
python train_cbm.py -dataset synthetic_2 -model_type joint -num_attributes 4 -num_classes 2 -seed 42 -epochs 50 --encoder_model inceptionv3 --weight_decay 0.004

echo "Small model"
python train_cbm.py -dataset synthetic_2 -model_type joint -num_attributes 4 -num_classes 2 -seed 42 -epochs 50 --encoder_model small3

echo "Regularized Small Model"
python train_cbm.py -dataset synthetic_2 -model_type joint -num_attributes 4 -num_classes 2 -seed 42 -epochs 50 --encoder_model small3 --weight_decay 0.004
