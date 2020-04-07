import pickle

if __name__ == '__main__':

    # Read path to training model. Ex : ../models/model.pkl
    model_path = input("Type in the path for the trained model: ")
    with open(model_path, 'rb') as file:
        mod = pickle.load(file)

    # Read path to input data. Ex : ../data/example_test_data.csv
    input_path = input("Type in the path for the input file: ")
    nb_suggestion = int(input("Type in the number of suggestions wanted: "))
    prediction = mod.predict(data_path=input_path, nb_suggestions=nb_suggestion)

    # Read path to save the predictions. Ex : ../processed/predictions.csv
    output_path = input("Type in path of the output file: ")
    prediction.to_csv(output_path, index=False)
