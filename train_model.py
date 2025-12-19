# ------------------------------------------------------------
# TRAIN MODEL FOR MUSIC GENERATION
# ------------------------------------------------------------
import os
import glob
import numpy as np
import pickle
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# ------------------------------------------------------------
# 1. Ensure "data" folder exists
# ------------------------------------------------------------
os.makedirs("data", exist_ok=True)

# ------------------------------------------------------------
# 2. Load MIDI Files
# ------------------------------------------------------------
notes = []

print("Loading MIDI files...")
for file in glob.glob("midi_songs/*.mid"):
    print("Parsing:", file)
    midi = converter.parse(file)

    try:
        parts = instrument.partitionByInstrument(midi)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes
    except:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

print("Total notes extracted:", len(notes))

# Save notes
with open("data/notes.pkl", "wb") as f:
    pickle.dump(notes, f)

# ------------------------------------------------------------
# 3. Prepare Sequences
# ------------------------------------------------------------
sequence_length = 100

pitchnames = sorted(set(notes))
note_to_int = {note: number for number, note in enumerate(pitchnames)}

network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]

    network_input.append([note_to_int[c] for c in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)
print("Total training patterns:", n_patterns)

network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
network_input = network_input / float(len(pitchnames))

network_output = to_categorical(network_output)

# ------------------------------------------------------------
# 4. Build LSTM Model
# ------------------------------------------------------------
model = Sequential([
    LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True),
    Dropout(0.3),
    LSTM(256, return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(256),
    Dropout(0.3),
    Dense(len(pitchnames), activation='softmax')
])

model.compile(loss="categorical_crossentropy", optimizer="adam")

# ------------------------------------------------------------
# 5. Save best model during training
# ------------------------------------------------------------
checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor="loss",
    verbose=1,
    save_best_only=True,
    mode="min"
)

# ------------------------------------------------------------
# 6. Train Model
# ------------------------------------------------------------
print("Training started...")

model.fit(
    network_input,
    network_output,
    epochs=50,
    batch_size=64,
    callbacks=[checkpoint]
)

model.save("music_model.h5")
print("Training completed and model saved!")
