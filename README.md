# CNN-Fake-Face-Recognition
Pipeline built for training models to recognize fake faces from real faces with a convolutional neural network based on Google's Inception V3. 

How to use: 
1. Download data with provided download scripts or use your own downloaded data. 
2. Run duplicate_deletion.py and delete_truncated_images.py on your data as instructed in the scripts.
3. Create dataset and model folders as instructed in binary_inception_v3.py. 
4. Use move_images.py to move your data into the newly created dataset folders. 
5. Run binary_inception_v3.py on your dataset for training and testing models. 
6. (Optional) Evaluate your training results using accuracy_evaluation.py. 
