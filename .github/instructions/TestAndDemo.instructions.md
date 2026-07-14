---
description: use this skill whenever you are required to read a file, study the concepts, and create a demo file to simulate the concepts that you have just learned
# applyTo: 'use this skill whenever you are required to read a file, study the concepts, and create a demo file to simulate the concepts that you have just learned' # when provided, instructions will automatically be added to the request context when the pattern matches an attached file
---

<!-- Tip: Use /create-instructions in chat to generate content with agent assistance -->

# important!
whenever I provide you a file to read, you read it carefully, understand the concepts presented in it, and then create a demo file to simulate the concepts that you have just learned. You should strictly follow the instructions provided in this file. Do not deviate from them.

# instructions for creating a demo file to simulate the concepts that you have just learned

1. Strictly follow the instructions provided in this file. Do not deviate from them.
2. Read the attached file carefully and understand the concepts presented in it.
3. Imitate the files that you have read as closely as possible
4. you should install any required packages that are necessary to run the code in the attached file, especially when the packages are used inside the files that you have just read
5. you should always use the real financial data downloaded from yfinance to simulate the concepts that you have just learned
6. you should not only present the data, but also plot graphs frequently to visualize the data and the results of your simulations
7. you must plot graph if the file that you have read contains any code that plots graphs
8. as this is for concepts learning purpose, you should make reference to concepts and explanation in the files that you have just read to write comments in your code, you graphs, and your output
9. be sure to make the graph seems neat, make low opacity lines if there are too much
10. adjust the graph to make them neat and clean based on the information you want to present
11. **IMPORTANT: External API handling** - If the source file uses yfinance or other external APIs to download data, ALWAYS ensure the graph is created. If the API call fails (rate limiting, network issues), either: (a) retry with a delay, (b) use alternative data sources, or (c) generate synthetic realistic data that simulates the concepts. DO NOT skip creating the graph—find an alternative way to generate it.

## CRITICAL: Graph Storage and Display Rules
12. **SAVE GRAPHS TO DISK ONLY** - When creating graphs with matplotlib:
    - ALWAYS use `plt.savefig()` to save graphs to the appropriate folder in the project
    - Save graphs with descriptive filenames in the same directory as the demo file
    - NEVER use `view_image()` or any other tool to display graphs in the chat
    - NEVER call `plt.show()` - it blocks execution and is not needed for automated demos
    - Confirm in terminal output that files were saved (e.g., "✓ Saved: filename.png")
    - The graphs are FOR THE USER to open and review themselves from their file system
    - Do NOT redirect the user to view graphs - just confirm they are saved




