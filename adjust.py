import numpy as np
import matplotlib.pyplot as plt
# Load the NumPy arrays


def MAPE(Y_Predicted,Y_actual):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape


stock_actuals = np.load('stock_actuals.npy')
stock_predictions = np.load('stock_predictions.npy')

print("Loaded stock_actuals shape:", stock_actuals.shape)
print("Loaded stock_predictions shape:", stock_predictions.shape)

true = []
pred = []
for i in range(stock_actuals.shape[0]) :
    true.append(stock_actuals[i][0])
    pred.append(stock_predictions[i][0])
true = np.array(true)
pred = np.array(pred)
error = print(MAPE(pred,true))

plt.figure(figsize=(30,10))

# Plot the true values in blue
plt.plot(true, label='True Values', color='blue', linestyle='-', marker='o')

# Plot the predicted values in red
plt.plot(pred, label='Predicted Values', color='red', linestyle='-', marker='x')

plt.ylabel('Stock Value')
plt.title('Asian Paints')

# Adding a legend
plt.legend()

# Save the figure
plt.savefig('asian paints.png')

# Display the plot
plt.show()
