import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

print("Testing matplotlib save...")
plt.figure()
plt.plot([1, 2, 3], [1, 4, 2])
plt.title('Test Plot')
output_path = r'c:\Users\kevin\Desktop\llm_debate\experiments\test_plot.png'
plt.savefig(output_path)
plt.close()
print(f"Saved to {output_path}")
print(f"File exists: {os.path.exists(output_path)}")
