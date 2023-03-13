# File: plot_train_hist.py
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Date: 03/13/2023
# Description: Plotting training history
# %%
import pandas as pd
import matplotlib.pyplot as plt
BATCH_SIZE = 32 # was 128
ALPHA = 1e-3 # was 1e-4
OUTPUT_DIR = f'/home/lawrence/conf_pred_under_attack/output/models/effnetv2b0_bs{BATCH_SIZE}_lr{ALPHA}'
df = pd.read_csv(f'{OUTPUT_DIR}/training_history.csv')

# %%
plt.figure(figsize=(18,8))

plt.suptitle('Training / Validation Loss & Accuracy History', fontsize=20)

plt.subplot(1,2,1)
plt.plot(df['loss'], label='Training Loss')
plt.plot(df['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Loss', fontsize=18)

plt.subplot(1,2,2)
plt.plot(df['accuracy'], label='Train Accuracy')
plt.plot(df['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.show()
# %%
