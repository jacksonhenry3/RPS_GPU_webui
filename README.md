# Running the RPS GPU WebUI

## Cluster Setup Instructions

1. **Request a GPU node:**
   ```sh
   sinteractive
   ```

2. **Load required modules:**
   ```sh
   module load uv
   module load cuda12.6
   ```

3. **Run the application:**
   ```sh
   uv run src/app.py
   ```

4. **Access the web interface:**
   - The server will print a URL in the terminal. Open it in your browser.