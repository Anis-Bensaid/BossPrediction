from BossPrediction import Model

if __name__ == '__main__':
    # Read path to training data. Ex : ../data/example_train_data.csv or ../data/data_20190804_test.csv
    data_path = input("\nType in the path for the training data: ")
    proxies_size = int(input("Type in the number of proxies to consider: "))
    mod = Model()
    mod.fit(data_path=data_path, proxies_size=proxies_size)
    # Read path to output data. Ex : ../models/model_25000.pkl
    output_path = input("\nType in the path for the output file (should end with .pkl): ")
    mod.save(output_path)