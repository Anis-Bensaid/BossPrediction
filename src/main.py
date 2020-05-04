import pickle

if __name__ == '__main__':
    # Read path to training model. Ex : ../models/model_25000.pkl
    model_path = input("Type in the path for the trained model (should end with .pkl): ")
    with open(model_path, 'rb') as file:
        mod = pickle.load(file)

    # Read path to input data. Ex : ../data/example_test_data.csv
    input_path = input("Type in the path for the input file: ")
    nb_suggestion = int(input("Type in the number of suggestions wanted: "))
    use_proxies = bool(input("Would you like to use proxies (Y/N): ").lower() == 'y')
    # Changes in the number of proxies used
    if use_proxies:
        change_proxies_param = bool(input("Currently the model uses " + str(mod.proxies_size) +
                                          " pairs to calculate proxies. " +
                                          "Would you like to change that ? (Y/N): ").lower() == 'y')
        if change_proxies_param:
            proxies_size = int(input("Type in the new number of pairs used to calculate : "))
        else:
            proxies_size = None
    prediction = mod.predict(data_path=input_path, nb_suggestions=nb_suggestion, use_proxies=use_proxies,
                             proxies_size=proxies_size)

    # Read path to save the predictions. Ex : ../processed/predictions.csv
    output_path = input("Type in path of the output file: ")
    prediction.to_csv(output_path, index=False)
