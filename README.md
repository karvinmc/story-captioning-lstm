# story-captioning-lstm

Story-driven image captioning using LSTM.

## Dataset Setup

The `dataset` folder is not included in this repository (see `.gitignore`).

To download the dataset from Kaggle:

1. **Download the dataset:**
   [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
2. **Check your folder structure:**
   - `dataset/Images/` (contains images)
   - `dataset/captions.txt` (contains captions)

---

## Installing Dependencies

Run these following commands:
```bash
pip install -r requirements.txt
```

## Training

Run the following command to train the model:

```bash
python train.py
```

## Notes

- The dataset is not tracked by git and must be downloaded manually.
- For more details, see the code and comments in `train.py`.
