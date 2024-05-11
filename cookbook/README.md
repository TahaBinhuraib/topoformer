# Topoformer Cookbook

example code and guides for accomplishing common tasks with Topoformer. To follow this guide, you'll need to `pip install` Topoformer first.

## Replicating the Small Scale IMDB Classification
Run the `train_imdb.py` file to replicate the small-scale IMDB sentiment classification training, testing, and visualization process mentioned in the paper.

## Extending the topoformer architecture to different modalities.
You can run `train_vit.py` to train a Vision Transformer (ViT) model on the CIFAR-10 dataset. This script demonstrates how to extend the Topoformer architecture to different modalities(image).


## Training Topoformer-BERT
To replicate the Topoformer-BERT model mentioned in the paper, follow the steps below:


1. cd into the `cramming_local` directory.
2. Follow the data installation steps outlined there(data in dropbox).
3. Run `python pretrain.py name=topoformer data=bookcorpus-wikipedia arch=bert-c5_topo train=bert-o3`

To evaluate on the GLUE benchmark, run the following command:

```python
python eval.py eval=GLUE_sane name=topoformer eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True
```