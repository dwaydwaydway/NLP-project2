echo train A
python3 f1_lossA.py 10 64
echo eval A
python3 Evaluate.py Data/labels-levela.csv ResultA/f1_SubmissionA_10_64.csv 
echo train B
python3 f1_lossB.py 10 64
echo eval B
python3 Evaluate.py Data/labels-levelb.csv ResultB/f1_SubmissionB_10_64.csv 
echo train C
python3 f1_lossC.py 10 64
echo eval C
python3 Evaluate.py Data/labels-levelc.csv ResultC/f1_SubmissionC_10_64.csv 
