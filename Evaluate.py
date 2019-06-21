import numpy as np
import pandas as pd
import sys
from sklearn.metrics import f1_score

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Format: python Evaluate.py Answer Result")
		exit()

	# Read in Result & Answer
	Answer = pd.read_csv(sys.argv[1], header=None)
	Result = pd.read_csv(sys.argv[2], header=None)

	# Check id are the same (which it should be)
	Diff = sum(Result[0] != Answer[0])
	if Diff == 0:
		print("All ID's are equal.")
	else:
		print("There are", Diff, "IDs that are not equal!!!")
		exit()

	# Calculate macro averaged f1 score
	y_true, y_pred = Answer[1], Result[1]
	Score = f1_score(y_true, y_pred, average='macro')

	print("Score:", Score)