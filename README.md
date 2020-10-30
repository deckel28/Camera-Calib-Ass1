# Camera-Calibration

## Requirements:
1. Numpy
2. Pandas
  
## Procedure:
Enter your dataset in [data/coords.csv]. Our dataset is already loaded in this file.

- Run [Camera_calibration.py](./Camera_calibration.py) which would start calibration using the Dataset provided in [data/coords.csv](./data/coords.csv). 
- It would print the Projection Matrix m and the Intrinsic and Extrinsic parameters.
- The recovered 2D points would be stored in [data/recovered_coords.csv](./data/recovered_coords.csv)
