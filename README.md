#### Character Detector for Devanagari Text

Devanagari, also called Nagari is an text used in India and Nepal. 
It is written from left to right, has a strong preference for symmetrical 
rounded shapes within squared outlines, 
and is recognisable by a horizontal line that runs along the top of full letters.
##### Dataset:
Dataset was obtained from: https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset

#### Procedure:
###### Pre-Processing:
The 32 by 32 images are vector converted by flattening into a vector of length 1024. 
###### Dimension Reduction:
The Data was tested using Principal Component Analysis. And a dimension reduction of 16 was decided.
###### Training:
Training was done on a Feed Forward Neural net with a hidden layer of 50 neurons.

The training accuracy as of now is 60%  
