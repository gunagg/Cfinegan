#!/bin/bash

echo "Going to generate Images"
python main_new.py --cfg cfg/eval.yml --gpu 2
echo "Images have been generated"

echo "Saving the feature vectors for original images"
python get_vectors_texture_orig.py

echo "Saving the feature vectors for generated images"
python get_vectors_texture_generated.py

echo "Saving the shape feature vectors for original images"
python get_vectors_shape_orig.py

echo "Saving the shape feature vectors for generated images"
python get_vectors_shape_generated.py


echo "comparing the images"
python compare_texture.py

echo "comparing the shape images"
python compare_shape.py

echo "completed"
