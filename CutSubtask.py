import sys
import pandas as pd
import numpy as np

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Format: python CutSubtask.py Data")
		exit()

	Data = pd.read_csv(sys.argv[1], sep="\t", header=0)

	# Split train data to 3 sets
	ColA = ['id', 'tweet', 'subtask_a']
	ColB = ['id', 'tweet', 'subtask_b']
	ColC = ['id', 'tweet', 'subtask_c']
	LevelA = Data[ColA]
	LevelB = Data[ColB]
	LevelC = Data[ColC]

	print("Level A:")
	print("\tNOT:", (LevelA.subtask_a == "NOT").sum())
	print("\tOFF", (LevelA.subtask_a == "OFF").sum())
	print("Level B:")
	print("\tTIN:", (LevelB.subtask_b == "TIN").sum())
	print("\tUNT", (LevelB.subtask_b == "UNT").sum())
	print("Level C:")
	print("\tIND:", (LevelC.subtask_c == "IND").sum())
	print("\tGRP", (LevelC.subtask_c == "GRP").sum())
	print("\tOTH", (LevelC.subtask_c == "OTH").sum())

	# Remove NULL in level b & c
	LevelB = LevelB[pd.notnull(LevelB.subtask_b)]
	LevelC = LevelC[pd.notnull(LevelC.subtask_c)]

	# Cut into different labels
	LevelA_NOT = LevelA[LevelA.subtask_a == "NOT"]
	LevelA_OFF = LevelA[LevelA.subtask_a == "OFF"]
	LevelB_TIN = LevelB[LevelB.subtask_b == "TIN"]
	LevelB_UNT = LevelB[LevelB.subtask_b == "UNT"]
	LevelC_IND = LevelC[LevelC.subtask_c == "IND"]
	LevelC_GRP = LevelC[LevelC.subtask_c == "GRP"]
	LevelC_OTH = LevelC[LevelC.subtask_c == "OTH"]
	
	# Save
	Header = ['id', 'tweet', 'subtask']
	LevelA.to_csv("Data/trainset-levela.tsv", sep='\t', index=0, header=Header)
	LevelB.to_csv("Data/trainset-levelb.tsv", sep='\t', index=0, header=Header)
	LevelC.to_csv("Data/trainset-levelc.tsv", sep='\t', index=0, header=Header)
	LevelA_NOT.to_csv("Data/trainset-levela-NOT.tsv", sep='\t', index=0, header=Header)
	LevelA_OFF.to_csv("Data/trainset-levela-OFF.tsv", sep='\t', index=0, header=Header)
	LevelB_TIN.to_csv("Data/trainset-levela-TIN.tsv", sep='\t', index=0, header=Header)
	LevelB_UNT.to_csv("Data/trainset-levela-UNT.tsv", sep='\t', index=0, header=Header)
	LevelC_IND.to_csv("Data/trainset-levela-IND.tsv", sep='\t', index=0, header=Header)
	LevelC_GRP.to_csv("Data/trainset-levela-GRP.tsv", sep='\t', index=0, header=Header)
	LevelC_OTH.to_csv("Data/trainset-levela-OTH.tsv", sep='\t', index=0, header=Header)