import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

print("Testing matplotlib...")

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
ax.set_title('Test Plot')

output_path = r'c:\Users\kevin\Desktop\llm_debate\experiments\test_matplotlib_output.png'
print(f"Saving to: {output_path}")

try:
    plt.savefig(output_path)
    plt.close()
    
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"SUCCESS! File created: {output_path}")
        print(f"File size: {size} bytes")
    else:
        print("FAILED! File was not created")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
