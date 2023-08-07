import matplotlib.pyplot as plt

# Assuming you have the values for loss, val_loss, categorical_accuracy, and val_categorical_accuracy
loss = transfer_model.model_history.history["loss"]
val_loss = transfer_model.model_history.history["val_loss"]
categorical_accuracy = transfer_model.model_history.history["categorical_accuracy"]
val_categorical_accuracy = transfer_model.model_history.history["val_categorical_accuracy"]

# Create a new figure
plt.figure(figsize=(12, 4))

# Plot loss and val_loss
plt.subplot(1, 2, 1)
plt.plot(loss, label="Training Loss", color='red', linestyle='--', linewidth=2)
plt.plot(val_loss, label="Validation Loss", color='blue', linewidth=2)
plt.xlabel("Iteration", fontweight='bold')  # Make x-axis label bold
plt.ylabel("Loss", fontweight='bold')  # Make y-axis label bold
plt.ylim(0,)
plt.title("Training and Validation Loss", fontweight='bold')  # Make title bold
plt.legend()
# Customize the appearance
plt.grid(True, linestyle='--', alpha=0.7)

# Plot categorical_accuracy and val_categorical_accuracy
plt.subplot(1, 2, 2)
plt.plot(categorical_accuracy, label="Training Accuracy", color='red', linestyle='--', linewidth=2)
plt.plot(val_categorical_accuracy, label="Validation Accuracy", color='blue', linewidth=2)
plt.xlabel("Iteration", fontweight='bold')  # Make x-axis label bold
plt.ylabel("Accuracy", fontweight='bold')  # Make y-axis label bold
plt.ylim(0,)
plt.title("Training and Validation Accuracy", fontweight='bold')  # Make title bold
plt.legend()

# Customize the appearance
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()
