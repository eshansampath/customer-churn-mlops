from src.train import train_model
from src.evaluate import evaluate
from src.explain import explain_model

def main():
    model, X_test, y_test, X_train, params = train_model()

    evaluate(model, X_test, y_test, X_train, params)

    explain_model(model, X_train) 

if __name__ == "__main__":
    main()