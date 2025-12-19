# Music-Generation-Using-Neural-Networks
This project demonstrates how deep learning can be used to generate music automatically. Using a Long Short-Term Memory (LSTM) neural network, the system learns musical patterns from existing MIDI files and generates new, original music compositions in MIDI format.


⚙️ Installation & Setup
Prerequisites

Python 3.8 or above

pip package manager

Install Dependencies
pip install music21 tensorflow keras numpy

📂 Project Structure


music_generation/


│


├── midi_songs/          # Input MIDI files


├── data/


│   └── notes.pkl        # Extracted notes


├── train_model.py       # Model training script


├── generate_music.py    # Music generation script


├── music_model.h5       # Trained model


└── generated_music.mid  # Output music file



▶️ How to Run the Project
Step 1: Add MIDI Files

Place multiple .mid files inside the midi_songs/ folder.

Step 2: Train the Model
python train_model.py

Step 3: Generate Music
python generate_music.py

Step 4: Play Output

Open generated_music.mid using any MIDI-supported music player.
