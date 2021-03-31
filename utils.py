import pretty_midi
import torch
import numpy as np


def numpy_to_midi(sample_roll, output='sample.mid'):
    music = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program(
        'Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    t = 0
    for i in sample_roll:
        if 'torch' in str(type(i)):
            pitch = int(i.max(0)[1])
        else:
            pitch = int(np.argmax(i))
        if pitch < 128:
            note = pretty_midi.Note(
                velocity=100, pitch=pitch, start=t, end=t + 1 / 8)
            t += 1 / 8
            piano.notes.append(note)
        elif pitch == 128:
            if len(piano.notes) > 0:
                note = piano.notes.pop()
            else:
                p = np.random.randint(60, 72)
                note = pretty_midi.Note(
                    velocity=100, pitch=int(p), start=0, end=t)
            note = pretty_midi.Note(
                velocity=100,
                pitch=note.pitch,
                start=note.start,
                end=note.end + 1 / 8)
            piano.notes.append(note)
            t += 1 / 8
        elif pitch == 129:
            t += 1 / 8
    music.instruments.append(piano)
    music.write(output)

def chord_to_midi(chord, output='chord.mid'):
    sample_roll = torch.zeros(32,130)
    sample_roll[np.where(chord==1)[0], np.where(chord[:, :]==1)[1]+60] = 1
    music = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program(
        'Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    t = 0
    count = 0
    for i in sample_roll:
        if count % 8 == 0:
            if 'torch' in str(type(i)):
                pitches = np.where(i==1)[0]
            else:
                pitches = np.where(i==1)[0]
            for pitch in pitches:
                if pitch < 128:
                    note = pretty_midi.Note(
                        velocity=100, pitch=pitch, start=t, end=t + 1)
                    piano.notes.append(note)
                elif pitch == 128:
                    if len(piano.notes) > 0:
                        note = piano.notes.pop()
                    else:
                        p = np.random.randint(60, 72)
                        note = pretty_midi.Note(
                            velocity=100, pitch=int(p), start=0, end=t)
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=note.pitch,
                        start=note.start,
                        end=note.end + 1)
                    piano.notes.append(note)

            t += 1
        count += 1
    music.instruments.append(piano)
    music.write(output)

def melody_to_numpy(fpath='samples/sally_melody_pitch.midi', unit_time=0.125, take_rhythm=False):
    music = pretty_midi.PrettyMIDI(fpath)
    notes = music.instruments[0].notes
    t = 0.
    roll = []
#     print(notes[0], notes[-1])
    for note in notes:
#         print(t, note)
        elapsed_time = note.start - t
        if elapsed_time > 0.:
            steps = torch.zeros((int(round(elapsed_time / unit_time)), 130))
            steps[range(int(round(elapsed_time / unit_time))), 129] += 1.
            roll.append(steps)
        n_units = int(round((note.end - note.start) / unit_time))
        steps = torch.zeros((n_units, 130))
        if take_rhythm:
            steps[0, 60] += 1
        else:
            steps[0, note.pitch] += 1
        steps[range(1, n_units), 128] += 1
        roll.append(steps)
        t = note.end
    return torch.cat(roll, 0)


def chord_to_numpy(fpath='samples/sally_chords.midi', unit_time=0.125):
    music = pretty_midi.PrettyMIDI(fpath)
    notes = music.instruments[0].notes
    max_end = 0.
#     print(notes)
    for note in notes:
#         print(note)
        if note.end > max_end:
            max_end = note.end
    chroma = torch.zeros((int(round(max_end / unit_time)), 12))
    for note in notes:
        idx = int(round((note.start / unit_time)))
        n_unit = int(round((note.end - note.start) / unit_time))
        chroma[idx:idx + n_unit, note.pitch % 12] += 1
    return chroma