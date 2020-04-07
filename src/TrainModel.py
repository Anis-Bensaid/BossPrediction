from BossPrediction import Model

if __name__ == '__main__':
    # Read path to training data. Ex : ../data/example_train_data.csv
    data_path = input("Type in the path for the training data: ")
    mod = Model()
    mod.fit(data_path=data_path)
    # Read path to output data. Ex : ../models/model.pkl
    output_path = input("Type in the path for the output file: ")
    mod.save(output_path)