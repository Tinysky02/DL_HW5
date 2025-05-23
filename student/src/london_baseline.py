# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.

import argparse
import utils

def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###
    with open("birth_dev.tsv" , encoding="utf-8") as f:
        n = sum(1 for _ in f)
    preds = ["London"] * n
    total, correct = utils.evaluate_places("birth_dev.tsv" , preds)
    accuracy = 0.0 if total == 0 else correct / total * 100.0 
    ### END YOUR CODE ###

    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
