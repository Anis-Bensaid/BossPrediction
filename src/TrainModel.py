from BossPrediction import Model

if __name__ == '__main__':
    # Read path to training data. Ex : ../data/
    data_path = input("Type in the path for the training data: ")
    mod = Model()
    mod.fit(data_path=data_path)
    output_path = input("Type in the path for the output file: ")
    mod.save(output_path)