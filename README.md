# CIGMO: Categorical invariant representations in a deep generative model 

This is the official pytorch implementation of CIGMO (Categorical Invariant Generative MOdel) by Haruo Hosoya [1].

# Dependencies
- pytorch 1.7.1
- torchvision 0.8.2
- numpy
- tqdm
- matplotlib
- scikit-image
- scikit-learn
- scipy

# Datasets
- ShapeNet [https://mega.nz/folder/FblQzB6J#_d4wyGwRv27xwqsBMLv-gA]
- MVC Cloth [https://mega.nz/folder/dT9DzTJY#LJmJIgUUaBJW3b1Tg-oIuw]

# Usage

To train one CIGMO instance with 3 categories (clusters) on ShapeNet with 3 object classes, type:

python -m run_shapenet --dataset_path=<path> --model_path=<path> --num_class=3 --num_cluster=3 --group_size=3 --gpu=0 --start_instance=0 --num_instance=1 --mode=train 

To evaluate the trained CIGMO model, type:

python -m run_shapenet --dataset_path=<path> --model_path=<path> --num_class=3 --num_cluster=3 --group_size=3 --gpu=0 --start_instance=0 --num_instance=1 --mode=test --save_result=True

For comparison, other types of models can be trained with changing options:
- GVAE: set --num_cluster=1
- Mixture of VAEs: set group_size=1
- MLVAE: add --mlvae=True

To train one IIC instance with 3 categories (clusters) on ShapeNet with 3 object classes, type:

python -m run_iic_shapenet --dataset_path=<path> --model_path=<path> --num_class=3 --num_cluster=3 --group_size=3 --gpu=0 --start_instance=0 --num_instance=1 --mode=train 

One can play with visualization etc. by running commands in nb_example_mvc.py or nb_example_shapenet.py, which are scripts for interactive use, e.g., Spyder.

# References and contact
If you publish a paper based on this code, please cite [1] or any following conference/journal publication.  If you have difficulty of downloading datasets, etc., please contact with the author.

[1] Haruo Hosoya.  CIGMO: Categorical invariant representations in a deep generative framework.  UAI 2022.

