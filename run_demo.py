import torch
import pretty_midi
import os
import numpy as np
import matplotlib.pyplot as plt
from model import VAE
from utils import *
import glob

def load_model(model_path):
    model = VAE(130, 2048, 3, 12, 128, 128, 32)
    model.eval()
    dic = torch.load(model_path)
    for name in list(dic.keys()):
        dic[name.replace('module.', '')] = dic.pop(name)
    model.load_state_dict(dic)
    return model

def swap(model, melody1, chord1, melody2, chord2, mode=1):
    melody1 = torch.stack(melody1.split(32, 0)).float()
    melody2 = torch.stack(melody2.split(32, 0)).float()
    chord1 = torch.stack(chord1.split(32, 0)).float()
    chord2 = torch.stack(chord2.split(32, 0)).float()
    
    if torch.cuda.is_available():
        model = model.cuda()
        melody1 = melody1.cuda()
        melody2 = melody2.cuda()
        chord1 = chord1.cuda()
        chord2 = chord2.cuda()
    with torch.no_grad():
        o_p, o_r = model.encoder(melody1, chord1)
        z1p, z1r = (o_p.mean, o_r.mean)
        r_p, r_r = model.encoder(melody2, chord2)
        z2p, z2r = (r_p.mean, r_r.mean)
        if mode == 1:
            test = model.decoder(z1p, z2r, chord1).cpu()
        if mode == 2:
            test = model.decoder(z1p, z1r, chord2).cpu()
    return test

def chord_to_ins(chord):
    chord = chord.cpu().numpy()
    ins = pretty_midi.Instrument(1)
    chords = []
    
    for i, c in enumerate(chord):
        note_index = tuple(np.where(c == 1)[0])
        if len(note_index) == 0:
            continue
        if len(chords) != 0 and chords[-1][0] == note_index:
            chords[-1][2] += 0.125
        else:
            chords.append([note_index, i * 0.125 , 0.125])
            
    notes = []
    for c in chords:
        start = c[1]
        end = c[1] + c[2]        
        appended = [pretty_midi.Note(80, p + 48, start, end) for p in c[0]]
        notes += appended
    ins.notes = notes
    return ins

def add_chord(melody_path, chord, target_path):
    midi = pretty_midi.PrettyMIDI(melody_path)
    midi.instruments.append(chord_to_ins(chord))
    midi.write(target_path)


def run(model_path):
    if not os.path.exists('demo'):
        os.mkdir('demo')
    model_name = os.path.join('demo', model_path.split('\\')[-1][0: -3])
    if not os.path.exists(model_name):
        os.mkdir(model_name)
        
    model = load_model(model_path)
    m1 = "original-melody/ashover12.mid"
    c1 = "original-chord/" + m1.split('/')[1]
    m2 = "original-melody/ashover13.mid"
    c2 = "original-chord/" + m2.split('/')[1]
    
    sec1 = 6.
    sec2 = 4.
    start1 = int(sec1 * 8)
    start2 = int(sec2 * 8)
    length = 1

    melody1 = melody_to_numpy(fpath=m1)[start1: start1 + 32 * length]
    chord1 = chord_to_numpy(fpath=c1)[start1: start1 + 32 * length]
    # print(chord1)
    melody2 = melody_to_numpy(fpath=m2)[start2: start2 + 32 * length]
    chord2 = chord_to_numpy(fpath=c2)[start2: start2 + 32 * length]
    
    chord1[chord1>1] = 1
    chord2[chord2>1] = 1
    
    # prepare 16th note C
    melody16 = torch.zeros(32, 130)
    melody16[:, 60] = 1.
    
    # prepare empty chord
    empty_chord = torch.zeros_like(chord1)
    
    # prepare scale melody
    scale = torch.zeros(32, 130)
    scale[[0, 3, 4, 8, 11, 12, 16, 18, 20, 
           21, 22, 23, 24, 25, 26, 27, 28], 
          [67, 71, 74, 71, 74, 79, 78, 76, 
           74, 76, 74, 72, 71, 72, 71, 69, 67]] = 1.
    scale[[1, 2, 5, 6, 7, 9, 10, 13, 14, 15, 17, 19, 29, 30, 31], 128] = 1.
    
    # prepare scale chord
    scale_chord = torch.zeros(32, 12)
    scale_chord[0: 16, [2, 7, 11]] = 1.
    scale_chord[16: 24, [2, 6, 9]] = 1.
    scale_chord[24: 32, [2, 7, 11]] = 1.
    
    # prepare shift one chord
    down_chord = torch.clone(chord1)
    down_chord = down_chord[:, list(range(1, 12)) + [0]]
    
    # prepare minor chord
    minor = torch.zeros(32, 12)
    minor[0: 16, [2, 7, 10]] = 1.
    minor[16: 24, [0, 3, 6, 9]] = 1.
    minor[24: 32, [2, 7, 10]] = 1.
    
    rhy1 = swap(model, melody1, chord1, melody16, empty_chord)
    rhy2 = swap(model, melody1, chord1, melody2, chord2)
    pit1 = swap(model, scale, scale_chord, melody1, chord1)
    pit2 = swap(model, melody2, chord2, melody1, chord1)
    mod1 = swap(model, melody1, chord1, melody1, down_chord, mode=2)
    mod2 = swap(model, melody1, chord1, melody1, minor, mode=2)
    
    items = [rhy1, rhy2, pit1, pit2, mod1, mod2, melody1, melody2, 
             melody2, scale, melody16]
    temp_names = ['rhy1', 'rhy2', 'pit1', 'pit2', 'mod1', 'mod2', 'original',
                  'pit2-ref', 'rhy2-ref', 'pit1-ref', 'rhy1-ref']
    chords = [chord1, chord1, scale_chord, chord2, down_chord, minor, chord1, 
              chord2, chord2, scale_chord, empty_chord]
    
    for i, (it, tn, chord) in enumerate(zip(items, temp_names, chords)):
        if i <= 5:
            numpy_to_midi(torch.cat(tuple(it), 0), output=tn + '.mid')
        else:
            numpy_to_midi(it, output=tn + '.mid')
        add_chord(tn + '.mid', chord, os.path.join(model_name, tn + '.mid'))
    

def batch_run():
    models = glob.glob('models/*.pt')
    for model_path in models:
        run(model_path)
    
    
    
#    numpy_to_midi(torch.cat(tuple(rhy1), 0), output='rhy1.mid')
#    numpy_to_midi(torch.cat(tuple(rhy2), 0), output='rhy2.mid')
#    numpy_to_midi(torch.cat(tuple(pit1), 0), output='pit1.mid')
#    numpy_to_midi(torch.cat(tuple(pit2), 0), output='pit2.mid')
#    numpy_to_midi(torch.cat(tuple(mod1), 0), output='mod1.mid')
#    numpy_to_midi(torch.cat(tuple(mod2), 0), output='mod2.mid')
#    numpy_to_midi(melody1, output='original.mid')
#    numpy_to_midi(melody2, output='pit2-ref.mid')
#    numpy_to_midi(melody2, output='rhy2-ref.mid')
#    numpy_to_midi(scale, output='pit1-ref.mid')
#    numpy_to_midi(melody16, output='rhy1-ref.mid')

    
        # numpy_to_midi(torch.cat())
    return 0
    

if __name__ == '__main__':
    batch_run()

#for i in range(len(empty_melody1)):
#    if i % 8 in [0, 2, 4, 5, 6, 7]:
#        empty_melody1[i, 62] = 1
#    else:
#        empty_melody1[i, 128] = 1


# for i in range(len(empty_melody1)):
#     if i % 8 in [0, 2, 3, 4, 5, 6]:
#         empty_melody1[i, 62] = 1
#     else:
#         empty_melody1[i, 128] = 1
        
# for i in range(len(empty_melody1)):
#     if i % 16 in [0, 3, 6, 8, 9, 14, 15]:
#         empty_melody1[i, 62] = 1
#     elif i % 8 in [1, 2, 4, 5, 7, 10, 11, 12, 13]:
#         empty_melody1[i, 129] = 1
#     else:
#         empty_melody1[i, 128] = 1

# chord2[chord2>1] = 1
# chord2 = chord1[:, list(range(1, 12)) + [0]]
# plot一下chord转化成12维后是啥样
# plt.matshow(np.flip(chord.transpose(0,1), 1), aspect='auto', origin='lower')
# print(melody1.shape, chord1.shape, empty_melody1.shape, empty_chord1.shape)






#if not os.path.isdir('generated'):
#    os.mkdir('generated')
#numpy_to_midi(torch.cat(tuple(test1), 0), output='generated/test1.mid')
#numpy_to_midi(torch.cat(tuple(test2), 0), output='generated/test2.mid')
#numpy_to_midi(torch.cat(tuple(test3), 0), output='generated/test3.mid')
#numpy_to_midi(torch.cat(tuple(test4), 0), output='generated/test4.mid')
#numpy_to_midi(torch.cat(tuple(test5), 0), output='generated/test5.mid')
#numpy_to_midi(torch.cat(tuple(test6), 0), output='generated/test6.mid')
#
#chord1_midi = pretty_midi.PrettyMIDI(c1)
#new1 = pretty_midi.PrettyMIDI("generated/test1.mid")
#for note in chord1_midi.instruments[0].notes:
#    note.start -= sec1
#    note.end -= sec1
#new1.instruments.append(chord1_midi.instruments[0])
#new1.write("generated/test1-complete.mid")
#
#chord2_midi = pretty_midi.PrettyMIDI(c2)
#new2 = pretty_midi.PrettyMIDI("generated/test2.mid")
#for note in chord2_midi.instruments[0].notes:
#    note.start -= sec2
#    note.end -= sec2
#new2.instruments.append(chord2_midi.instruments[0])
#new2.write("generated/test2-complete.mid")
#
#chord2_midi = pretty_midi.PrettyMIDI(c2)
#new3 = pretty_midi.PrettyMIDI("generated/test3.mid")
#for note in chord2_midi.instruments[0].notes:
#    note.start -= sec2
#    note.end -= sec2
#new3.instruments.append(chord2_midi.instruments[0])
#new3.write("generated/test3-complete.mid")
#
#chord1_midi = pretty_midi.PrettyMIDI(c1)
#new4 = pretty_midi.PrettyMIDI("generated/test4.mid")
#for note in chord1_midi.instruments[0].notes:
#    note.start -= sec1
#    note.end -= sec1
#new4.instruments.append(chord1_midi.instruments[0])
#new4.write("generated/test4-complete.mid")
#
#chord2_midi = pretty_midi.PrettyMIDI(c2)
#new5 = pretty_midi.PrettyMIDI("generated/test5.mid")
#for note in chord2_midi.instruments[0].notes:
#    note.start -= sec2
#    note.end -= sec2
#new5.instruments.append(chord2_midi.instruments[0])
#new5.write("generated/test5-complete.mid")
#
#chord1_midi = pretty_midi.PrettyMIDI(c1)
#new6 = pretty_midi.PrettyMIDI("generated/test6.mid")
#for note in chord1_midi.instruments[0].notes:
#    note.start -= sec1
#    note.end -= sec1
#new6.instruments.append(chord1_midi.instruments[0])
#new6.write("generated/test6-complete.mid")