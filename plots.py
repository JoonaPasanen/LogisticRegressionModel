import matplotlib.pyplot as plt
from process_data import get_data, mask_data, compute_covariates
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, brier_score_loss, log_loss

def main():
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
    accuracy = model.score(scaled_testX, testY)
    print("log loss:", log_loss(testY, predictions))
    print("Brier score:", brier_score_loss(testY, predictions))
    print("Accuracy:", accuracy)

    # Plotting results    
    probs = predictions[:, 1]
    fig, axs = plt.subplots(1, 3, figsize=(10, 8))

    # Plot 1: ROC Curve

    # NOTE: ROC Curve plots the relation between True Positives vs. False Positives
    #       using a set of different "threshold values" meaning that every
    #       prediction probability over the threshold is considered a positive.
    #       The straight line shows the case where the predicted probabilities
    #       are drawn randomly from the range [0, 1].
    
    fpr, tpr, _ = roc_curve(testY, probs)
    auc = roc_auc_score(testY, probs)
    axs[0].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    axs[0].plot([0, 1], [0, 1], linestyle="--", label=f"Random = 0.500")
    axs[0].grid(color='lightgray', linestyle='--', linewidth=0.7)
    axs[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    axs[0].set_title("ROC Curve")
    axs[0].legend()

    # Plot 2: Precission-Recall Curve

    # NOTE: Precission-Recall Curve plots the relation between precision and recall.
    #       Precission is defined as the number of true positives over all the number
    #       of all positive classifications. Recall on the other hand is defined as the number
    #       of true positives over the number of true positives plus the number of false negatives.
    #       The plot thus captures the relationship between the accuracy of the models positive predictions
    #       and the rate at which it "captures" positive values. This is done with a number of diffrent thresholds.
    
    p, r, _ = precision_recall_curve(testY, probs)
    axs[1].plot(r, p)
    axs[1].grid(color='lightgray', linestyle='--', linewidth=0.7)
    axs[1].set(xlabel="Recall", ylabel="Precision")
    axs[1].set_title("Precision-Recall Curve")

    # Plot 3: Confusion Matrix

    # NOTE: Confusion Matrix consisting of entries C_ij represents
    #       the values that are known to be in group i, and were predicted
    #       to be in group j. So for example C_11 represents games where player A won
    #       and the model predicted player A to win. The rows of the matrix are normalized
    #       such that each row must sum to 1. Meaning that each entry on a row can be thought of
    #       as the probability that the value i gets predicted as the value j.

    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(testY, preds, normalize="true")
    im = axs[2].imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=axs[2])
    axs[2].set(
        xticks=[0,1], 
        yticks=[0,1], 
        xlabel="Predicted", 
        ylabel="Actual",
        title="Confusion Matrix"
    )
    for i in [0, 1]:
        for j in [0, 1]:
            color = "white" if cm[i, j] > 0.5 else "black"
            axs[2].text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color=color)

    plt.suptitle("Model: Logistic Regression")
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()

if __name__ == "__main__":
    main()