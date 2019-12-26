# Cfinegan

## Code of our paper "cFineGAN: Unsupervised multi-conditional fine-grained image generation" accepted at NeurIPS Workshop on Machine Learning for Creativity and Design 3.0. Paper link [here](https://arxiv.org/abs/1912.05028)

Run the following to train FineGAN over the dataset of interest 
```
cd code
python main.py --cfg cfg/train.yml --gpu 0
```

To perform condititonal generation, the following steps need to be done - a) Store all the possible generated images b) Compute the texture feature vector of all generated and real images c) Compute the shape feature vector of all generated and real images d) Compute the nearest neighbours in the embedding space of the conditional inputs . 
The following script runs all the above steps -
```
bash run_all.sh
```

