# FIFA 18 Value Prediction

This project predicts the players *value* based on [*FIFA 18 dataset*](https://www.kaggle.com/thec03u5/fifa-18-demo-player-dataset).

For example, *Lionel Messi* FIFA attributes:

![Lionel Messi](images/lionel_messi.jpg?raw=true "Lionel Messi")

Also includes a simple *Flask* application that can serve predictions from a model (*Linear Regression Model*). Reads a pickled model (*/rest_api/lin_reg_model.pkl*) into memory when the *Flask application* is started and returns predictions through the */predict* endpoint.

## API REST

*/predict* (GET) returns a prediction.

Optional parameters: ["Overall", "Potential", "Wage", "Special", "Ball control", "Composure", "Reactions", "Short passing", "CAM", "CF", "CM", "LAM", "LCM", "LM", "LS", "RAM", "RCM", "RM", "RS", "ST"]

*Default value for not included parameters: 0*.

For example:

```
localhost:8080/predict?Overall=85&Potential=80&Ball control=75
```

Returns:

```
{
    "value_prediction": "€2,278,889.11"
}
```

## Project files
*  *data_exploration_cleaning.ipynb*: Data exploration and cleaning
*  *data_modeling.yml*: Data modeling


## Data
*  */data/raw/CompleteDataset.csv*: Complete player dataset
*  */data/processed/CompleteDataset_cleaned.csv*: Complete player dataset cleaned
*  */data/output/predictions.csv*: Model predictions

## Flask Application
*  */rest_api/app.py*: Flask application
*  */rest_api/lin_reg_model.pkl*: Pickled Linear Regression Model

## Sources
  * [FIFA 18 Complete Player Dataset (Kaggle)](https://www.kaggle.com/thec03u5/fifa-18-demo-player-dataset)
  * [Fifa 18 Transfer Value-Wage Model (Kaggle)](https://www.kaggle.com/fournierp/fifa-18-transfer-value-wage-model)
  * [Player Wage Prediction (Kaggle)](https://www.kaggle.com/stahamtan/player-wage-prediction)
  * [SoFIFA.com](https://sofifa.com/)