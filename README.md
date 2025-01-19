## Features

- Text Classification using audio file.
- User-friendly web interface.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x installed on your local machine.
- Python packages listed in the `requirements.txt` file.
- Virtualenv installed (you can install it using `pip install virtualenv`).

## Installation

1. Clone the repository:

   ```bash
   git clone [link to github repo]
   ```

2. Navigate to the project directory:

   ```bash
   cd [directory name]
   ```

3. Create a virtual environment (replace `venv` with your preferred name):

   ```bash
   virtualenv venv
   ```

4. Activate the virtual environment:

   - On Windows:

   ```bash
   venv\Scripts\activate
   ```

   - On macOS and Linux:

   ```bash
   source venv/bin/activate
   ```

5. Install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask application:

   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://localhost:5000`.

3. Upload an audio file containing data for classification.

4. After processing, you can see the results.

## Configuration

You can customize the behavior of the app by modifying the `config.py` file.
