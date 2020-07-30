# DL-sentiment-analysis

### Movie Reviews
We retrieved the movie reviews from http://ai.stanford.edu/~amaas/data/sentiment/.
We processed them and created a `train.csv` and `test.csv` which can be found in the `/data` folder for easier usage.

### Setup
The project uses `Python >=3.6` in general and `torch` as machine learning library. Additional necessary libraries can
be found in `requirements.txt`. We recommend to create a virtual environment and install them as follows:

```
pip install -r requirements.txt 
```

### Training
```
python train.py
```

| Parameter          | Default          | Description                                                                  |
|--------------------|------------------|------------------------------------------------------------------------------|
| --train-data-path  | `data/train.csv` | File location where the processed training data is stored.                   |
| --model-checkpoint | `model.pt`       | File location where the trained model will be stored.                        |
| --vocab-checkpoint | `checkpoints/vocab.txt` | File location where the vocabulary of the sequence tokenizer will be stored. |
| --grid-search      | `False`          | If `True` performs grid-search to find the best model configuration.         |
| --num-epochs       | `50`             | The number of epochs for training.                                           |
| --patience         | `5`              | The number of epochs early-stopping is waiting for significant changes.      |


### Testing
```
python test.py --model-checkpoint model.pt
```
| Parameter          | Default          | Description                                                                           |
|--------------------|------------------|---------------------------------------------------------------------------------------|
| --test-data-path   | `data/test.csv`  | File location where the processed test data is stored.                                |
| --model-checkpoint | `model.pt`       | File location where the model from training is stored.                                |
| --vocab-checkpoint | `checkpoints/vocab.txt` | File location where the vocabulary of the sequence tokenizer from training is stored. |

### Write your own review
```
python write_review.py --model-checkpoint model.pt
```
| Parameter          | Default          | Description                                                                           |
|--------------------|------------------|---------------------------------------------------------------------------------------|
| --model-checkpoint | `model.pt`       | File location where the model from training is stored.                                |
| --vocab-checkpoint | `checkpoints/vocab.txt` | File location where the vocabulary of the sequence tokenizer from training is stored. |
