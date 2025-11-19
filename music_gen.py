import glob
import numpy as np
import pickle
from music21 import converter, instrument, note, chord, stream

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical


# ------------------------------------------------------------
# STEP 1: LOAD DATASET (Nottingham MIDI)
# ------------------------------------------------------------
def load_notes():
    notes = []

    # Read all MIDI files in /midi folder
    for file in glob.glob("midi/*.mid"):
        print("Parsing:", file)
        midi = converter.parse(file)

        try:
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    print("Total notes extracted:", len(notes))

    with open("notes.pkl", "wb") as f:
        pickle.dump(notes, f)

    return notes


# ------------------------------------------------------------
# STEP 2: PREPARE SEQUENCES FOR TRAINING
# ------------------------------------------------------------
def prepare_sequences(notes):
    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)

    note_to_int = {note: i for i, note in enumerate(pitchnames)}

    seq_length = 100
    network_input = []
    network_output = []

    for i in range(0, len(notes) - seq_length):
        seq_in = notes[i:i + seq_length]
        seq_out = notes[i + seq_length]

        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    print("Total sequences:", n_patterns)

    # reshape + normalize input
    network_input = np.reshape(network_input, (n_patterns, seq_length, 1))
    network_input = network_input / float(n_vocab)

    # one-hot output
    network_output = to_categorical(network_output)

    return network_input, network_output, n_vocab, pitchnames


# ------------------------------------------------------------
# STEP 3: BUILD LSTM MODEL
# ------------------------------------------------------------
def build_model(network_input, n_vocab):
    model = Sequential([
        LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True),
        Dropout(0.3),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dense(256),
        Dropout(0.3),
        Dense(n_vocab),
        Activation("softmax")
    ])

    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


# ------------------------------------------------------------
# STEP 4: TRAIN & SAVE MODEL
# ------------------------------------------------------------
def train():
    notes = load_notes()
    network_input, network_output, n_vocab, pitchnames = prepare_sequences(notes)

    with open("pitchnames.pkl", "wb") as f:
        pickle.dump(pitchnames, f)

    model = build_model(network_input, n_vocab)

    model.fit(network_input, network_output, epochs=50, batch_size=64)
    model.save("music_model.h5")

    print("Training done!")


# ------------------------------------------------------------
# STEP 5: GENERATE MUSIC
# ------------------------------------------------------------
def generate():
    with open("notes.pkl", "rb") as f:
        notes = pickle.load(f)

    with open("pitchnames.pkl", "rb") as f:
        pitchnames = pickle.load(f)

    n_vocab = len(pitchnames)

    int_to_note = {i: n for i, n in enumerate(pitchnames)}
    note_to_int = {n: i for i, n in enumerate(pitchnames)}

    model = build_model(np.zeros((1, 100, 1)), n_vocab)
    model.load_weights("music_model.h5")

    # Pick a random starting point
    start = np.random.randint(0, len(notes) - 100)
    pattern = [note_to_int[n] for n in notes[start:start + 100]]

    output_notes = []

    for _ in range(500):
        input_seq = np.reshape(pattern, (1, len(pattern), 1))
        input_seq = input_seq / float(n_vocab)

        prediction = np.argmax(model.predict(input_seq, verbose=0))

        result = int_to_note[prediction]
        output_notes.append(result)

        pattern.append(prediction)
        pattern = pattern[1:]

    create_midi(output_notes)


# ------------------------------------------------------------
# STEP 6: WRITE OUTPUT MIDI FILE
# ------------------------------------------------------------
def create_midi(prediction_output):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if "." in pattern:
            notes_in_chord = pattern.split(".")
            notes_list = [note.Note(int(n)) for n in notes_in_chord]
            new_chord = chord.Chord(notes_list)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp="generated/generated_music.mid")

    print("Music generated → generated/generated_music.mid")


# ------------------------------------------------------------
# RUNNER
# ------------------------------------------------------------
if __name__ == "__main__":
    print("1: Train model")
    print("2: Generate music")
    ch = input("Enter choice: ")

    if ch == "1":
        train()
    else:
        generate()