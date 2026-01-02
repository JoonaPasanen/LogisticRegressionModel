from process_data import get_data, mask_data, compute_covariates
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss

def logistic_regression():
    df = get_data(2000, 2024)
    df = mask_data(df, seed=1)
    df = compute_covariates(df)

    train_data = df[~df["tourney_date"].astype(str).str.startswith("2024")]
    test_data = df[df["tourney_date"].astype(str).str.startswith("2024")]

    scaler = StandardScaler()

    y = train_data["winner"]    
    X = train_data.drop(["tourney_date", "winner", "surface"], axis="columns")
    scaled_X = scaler.fit_transform(X)

    model = LogisticRegression().fit(scaled_X, y)

    testY = test_data["winner"]
    testX = test_data.drop(["tourney_date", "winner", "surface"], axis="columns")
    scaled_testX = scaler.transform(testX)

    predictions = model.predict_proba(scaled_testX)
    print("log loss:", log_loss(testY, predictions))
    print("Brier score:", brier_score_loss(testY, predictions))

if __name__ == "__main__":
    logistic_regression()