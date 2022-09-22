# RGTN-NIE

Dataset and code for Representation Learning on Knowledge Graphs for Node Importance Estimation.

## NIE Dataset

* FB15k: a subset from [FreeBase](https://developers.google.com/freebase).

* TMDB5k: original files are from 
[Kaggle](https://www.kaggle.com/tmdb/tmdb-movie-metadata).

* IMDB: original files are from [IMDb Datasets](
https://www.imdb.com/interfaces/).
We provide the node text description files on [Google Drive](https://drive.google.com/file/d/10y6yIN6_y1Mw_83RKP32KISql_INjrWK/view?usp=sharing), and the graph construction files on [Google Drive](
https://drive.google.com/file/d/1xd0ObAIDYMsxQZD2l0e-9fWo_ro76--x/view?usp=sharing).

* Processed features: [Google Drive](https://drive.google.com/drive/folders/1mgcNhGHUTptTqRREJE-g-qKoGycVwKpV?usp=sharing).
Download the feature files and put them on 'datasets'.

## Dependencies 
* pytorch 1.6.0
* DGL 0.5.3

## Training Examples

* run `sh train_geni.sh` for GENI in FB15k (full batch training)
* run `sh train_geni_batch.sh` for GENI in IMDB (minibatch training)
* run `sh train_two.sh` for RGTN in FB15k (full batch training)
* run `sh train_two_batch.sh` for RGTN in IMDB (minibatch training)

Note that hyperparameters may require grid search in small datasets.


## Citation
If you find our work useful for your reseach, please consider citing this paper:
```bibtex
@inproceedings{Huang21RGTN-NIE,
  author    = {Han Huang and Leilei Sun and Bowen Du and Chuanren Liu and Weifeng Lv and Hui Xiong},
  title     = {Representation Learning on Knowledge Graphs for Node Importance Estimation},
  booktitle = {{KDD} '21: The 27th {ACM} {SIGKDD} Conference on Knowledge Discovery and Data Mining, Virtual Event, Singapore, August 14-18, 2021},
  pages     = {646--655},
  publisher = {{ACM}},
  year      = {2021}
}
```
