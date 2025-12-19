import pickle
import numpy as np
from music21 import instrument, note, stream, chord
from tensorflow.keras.models import load_model

# Load saved data
with open('data/notes.pkl', 'rb') as f:
    notes = pickle.load(f)

pitchnames = sorted(set(notes))
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

model = load_model("music_model.h5")

sequence_length = 100
network_input = []

# Prepare seed pattern
for i in range(0, len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    network_input.append([note_to_int[c] for c in seq_in])

pattern = network_input[np.random.randint(0, len(network_input) - 1)]

# Generate notes
prediction_output = []

for note_index in range(300):  # generate 300 notes
    input_seq = np.reshape(pattern, (1, len(pattern), 1))
    input_seq = input_seq / float(len(pitchnames))

    prediction = model.predict(input_seq, verbose=0)
    index = np.argmax(prediction)
    result = pitchnames[index]

    prediction_output.append(result)

    pattern.append(index)
    pattern = pattern[1:]

# Convert predictions to midi
output_notes = []

for pattern in prediction_output:
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes_objs = [note.Note(int(n)) for n in notes_in_chord]
        new_chord = chord.Chord(notes_objs)
        new_chord.duration.quarterLength = 0.5
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.duration.quarterLength = 0.5
        output_notes.append(new_note)

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='generated_music.mid')

print("Music generated and saved as generated_music.mid")
