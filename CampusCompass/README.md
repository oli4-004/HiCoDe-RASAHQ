# codespaces-quickstart
Get started with Rasa in the browser using GitHub Codespaces.
Please note that our entire project resides in the CampusCompass folder. Please change directory if you are in the parent folder HiCoDe-RASAHQ.

### Steps

1. **Create a Codespace:**
   - Click on the green "Code" button on this page, then scroll down to "Codespaces".
   - Click on "Create codespace on main".

2. **Set Up Environment:**
   - In the codespace, open (or create if not existing) the `.env` file from this repo and add the RASA, OpenAI and Google maps API keys. Use a new line for each API key
     ```
     RASA_LICENSE='your_key_here'
     OPENAI_API_KEY='your_key_here'
     GOOGLE_MAPS_API_KEY='your_key_here'
     ```
   - Set this environment variables by running in the terminal
     ```
     source .env
     ```
     Navigate to the Campus Compass directory by running
     ```
     cd [COPY CAMPUSCOMPASS FOLDER AS PATH AND PASTE HERE]
     ```
     Install uv
     ```
     pip install uv
     ```
     Python 3.11 is required so to install it in the virtual environment run
     ```    
     uv venv --python 3.11
     ```
     Activate your python environment by running
     ```
     .venv\Scripts\activate
     ```
     Lastly, to install the dependencies run
     ```
     uv pip install -r requirements.txt

4. **Train and run the Model:**
   - In the terminal, run:
     ```
     .\run-and-train-dev.ps1
     ```
     The bot should now open in the default browser

5. **Talk to your Bot:**
   - If training is not needed you can run in the terminal
     ```
     .\run-dev.ps1
     ```
     The bot should now open in the default browser