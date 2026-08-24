# Earthquake Risk Prediction using LSTM, GRU, and RNN

This project tries to predict if a future earthquake will be **high risk** (magnitude 6.0 or higher) by looking at patterns in past earthquakes. It uses and compares three deep learning models that are good at understanding sequences over time: **LSTM, GRU, and RNN**.

## What This Project Does

Earthquakes don't happen randomly out of nowhere — they happen in a sequence, one after another, over time. This project treats earthquake history like a time series (similar to stock prices or weather) and asks:

> "Looking at the last 30 earthquakes, can we guess if the next one will be strong (magnitude 6.0+) or normal?"

The project explores this question step by step, starting simple and getting more advanced, and honestly reports what worked and what didn't.

## Dataset

The dataset used is `database.csv`, which contains **23,412 earthquake records** with 21 columns of information. This project mainly uses:
- **Date & Time** — when the earthquake happened
- **Latitude & Longitude** — where it happened
- **Depth** — how deep underground it occurred (in km)
- **Magnitude** — how strong it was

> Note: The dataset file is not included in this repository because it's too large for GitHub. You can download a similar dataset from [Kaggle - Earthquake Database](https://www.kaggle.com/datasets/usgs/earthquake-database) and place it in the project folder as `database.csv.zip` before running the notebook.

## Step-by-Step Process

### 1. Load and Clean the Data
- Extracted the zip file and loaded `database.csv`.
- Checked the shape (23,412 rows, 21 columns) and checked which columns had missing values.
- Kept only the useful columns: Date, Time, Latitude, Longitude, Depth, Magnitude.
- Removed rows where Magnitude was missing (since Magnitude is what we're predicting).

### 2. Fix the Dates
- Converted the Date column to a proper date format.
- Some dates had mixed time zones, which caused a warning — fixed by converting everything to UTC (a single, standard time zone).
- Sorted all rows from oldest to newest date, because in time series data, the order matters a lot.

### 3. Scale the Numbers
- Used MinMaxScaler to squeeze all numeric values (magnitude, depth, latitude, longitude) into a 0–1 range. This helps deep learning models learn faster and more evenly.

### 4. Turn Data into Sequences
- Built a function that takes the past **30 earthquakes** as input and the **31st earthquake** as the answer to predict (this is called a "sliding window" of size 30).
- Split the data into 80% for training and 20% for testing, keeping the time order (not shuffled, since future data shouldn't leak into training).

## Experiment 1: Predicting the Exact Magnitude Number (Regression)

First, the project tried to predict the **exact magnitude value** (like 5.8, 6.2, etc.) using only the Magnitude column from the past 30 earthquakes.

- Built an **LSTM** model (50 units) and a **GRU** model (50 units), each with a dropout layer to avoid overfitting.
- Both were trained for 20 epochs.
- Then improved this by using **4 features** instead of just 1 (Magnitude, Depth, Latitude, Longitude) to give the model more context.
- Plotted actual vs predicted magnitude values on a graph to see how close the predictions were.

**Finding:** The model could follow the general pattern, but predicting the *exact* magnitude number turned out to be very hard — earthquake magnitudes don't follow a clean, learnable pattern from history alone.

## Experiment 2: Predicting High Risk vs Normal (Classification)

Since predicting the exact number was too hard, the project switched to a simpler, more useful question: **will the next earthquake be high risk (magnitude 6.0+) or not?**

- Created a new column `Risk`: 1 = high risk (magnitude ≥ 6.0), 0 = normal.
- Checked the balance: **16,058 normal** earthquakes vs **7,354 high-risk** ones — so high-risk cases are the minority (imbalanced data).
- Built an LSTM classifier (with a sigmoid output, which gives a probability between 0 and 1).

**First result (no balancing):** The model just predicted "normal" for almost everything, catching **0% of high-risk earthquakes**. This is a common problem when one class is much bigger than the other — the model takes the "lazy" shortcut.

## Experiment 3: Fixing the Imbalance (Class Weighting)

- Used `compute_class_weight` to tell the model "pay more attention to the high-risk class" (roughly 1.6x more weight on class 1).
- Retrained the LSTM model with these class weights.
- **Result:** Precision 0.33 / Recall 0.27 for high-risk earthquakes — better than before, but recall was still low.

## Experiment 4: Trying Different Decision Thresholds

By default, a model says "high risk" only if its probability is above 0.5. The project tried different cut-off points to catch more high-risk earthquakes:

| Threshold | What Happened |
|-----------|---------------|
| 0.35 | Caught 100% of high-risk cases, but flagged almost everything as risky (not useful — too many false alarms) |
| 0.45 (best F1 threshold, found automatically) | Precision 0.32, Recall 0.999 — same problem, too many false alarms |
| 0.50 (default) | Precision 0.33, Recall 0.27 |
| 0.55 | Precision 0.36, Recall 0.03 — barely catches any high-risk cases |
| 0.58 | Precision 0.39, Recall 0.01 — even fewer caught |

**Finding:** There's a trade-off — a lower threshold catches more real high-risk earthquakes but also raises many false alarms, while a higher threshold is more "confident" but misses most real high-risk cases. Also plotted a histogram of the model's predicted probabilities to see how confident (or unsure) it generally was.

## Experiment 5: Comparing RNN, GRU, and LSTM

All three models were trained the same way (same features, same class weighting) to see which sequence model handles this problem best:

| Model | Precision (high-risk) | Recall (high-risk) | F1-score |
|-------|------------------------|---------------------|----------|
| RNN   | 0.31                   | 0.54                | 0.40     |
| LSTM  | 0.32                   | 0.58                | 0.41     |
| GRU   | 0.32                   | 0.46                | 0.38     |

**LSTM came out on top**, catching 58% of the true high-risk earthquakes in the test data.

## Experiment 6: Adding a "Time Gap" Feature

- Added a new feature: the number of days since the last earthquake (`TimeGap`).
- Rebuilt the dataset with **5 features** (Magnitude, Depth, Latitude, Longitude, TimeGap) and retrained the LSTM model.
- **Result:** Precision 0.36, Recall 0.05 — this was actually *worse* than the 4-feature version, especially for catching real high-risk cases.

**Finding:** Adding the time gap between earthquakes did not help the model — in fact, it hurt performance.

## Experiment 7: Focusing on Specific Regions

Tried narrowing the data down to earthquake-prone regions to see if a more focused dataset would help:

- **Japan region** (filtered by latitude/longitude box): 1,780 earthquakes found — too small a dataset to train on properly.
- **Pacific "Ring of Fire"** (a much larger active zone covering places like Japan, Philippines, Indonesia, Alaska, California, and Chile): 18,543 earthquakes found, with 5,881 of them high-risk.
- Retrained the LSTM classifier only on Ring of Fire data.
- **Result:** Precision 0.32, Recall 0.41 — still not better than using the full worldwide dataset.

**Finding:** Focusing on a specific active region did not improve results either.

## Final Results and Conclusion

| Model | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| RNN   | 0.31      | 0.54   | 0.40     |
| LSTM  | 0.32      | 0.58   | 0.41     |
| GRU   | 0.32      | 0.46   | 0.38     |

The **LSTM model with 4 features** (magnitude, depth, latitude, longitude) and **class weighting** performed best overall, correctly identifying **58% of high-risk earthquakes** in the test set.

Extra experiments — adding a time-gap feature, and narrowing the data to specific regions — did **not** improve results. This is an honest finding, not a failure: it shows that predicting earthquake risk from historical patterns alone has real, well-known limits, which matches what actual seismologists have found in real research. Earthquakes are influenced by deep geological factors (like tectonic stress underground) that simple historical patterns can't fully capture.

## Tools and Libraries Used

- Python
- Pandas & NumPy — for handling and processing data
- Scikit-learn — for scaling data, class weighting, and evaluating results (confusion matrix, precision/recall, F1-score)
- TensorFlow / Keras — for building the LSTM, GRU, and RNN models
- Matplotlib — for plotting graphs (actual vs predicted, probability distribution)

## How to Run This Project

1. Download the dataset and place `database.csv.zip` in the project folder.
2. Install the required libraries:
   ```
   pip install pandas numpy scikit-learn tensorflow matplotlib
   ```
3. Open the notebook file (`.ipynb`) in Jupyter Notebook or Google Colab.
4. Run the cells in order from top to bottom.

## What I Learned

- How to prepare time series data for deep learning using sliding windows.
- The difference between predicting an exact number (regression) and predicting a category (classification), and why classification worked better here.
- How to handle imbalanced classes using class weighting.
- How changing the decision threshold changes the trade-off between catching more true cases and avoiding false alarms.
- How LSTM, GRU, and RNN compare on the same real-world task.
- Why negative results (features or filtering that don't help) are still valuable — they show what genuinely doesn't work and why, which is how real research operates.
