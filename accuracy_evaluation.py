## Evaluate accuracy of the model from a log file

import re
import statistics

# The path to the models folder containing its log file.
model_path = "./Trained_Models/Pretrained_50epochs_TpDne/"

# Load the model
log_file = open(model_path + ".log.txt", "r")

# Get training and validation accuracies from the log file
accuracy_str = ""
train_accuracy = []
val_accuracy = []
for line in log_file:
    accuracy_str += line
accuracy_str = re.sub(r'[^0-9. ]', '', accuracy_str)
accuracy_str = accuracy_str.split(' ')
to_remove = []
for i in range(len(accuracy_str)):
    if accuracy_str[i] == '':
        to_remove.append(i)
for i in range(len(to_remove)):
    accuracy_str.pop(to_remove[i]-i)
for i in range(len(accuracy_str)):
    if i % 5 == 2:
        train_accuracy.append(float(accuracy_str[i]))
    elif i % 5 == 4:
        val_accuracy.append(float(accuracy_str[i]))

# Do whatever calculations needed here
last_50_train_accuracy = train_accuracy[-50:]
print(statistics.mean(val_accuracy))

log_file.close()
