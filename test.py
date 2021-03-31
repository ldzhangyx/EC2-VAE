import torch
import pretty_midi
import os
import numpy as np
import matplotlib.pyplot as plt
from attn_ecvae_mq import VAE
from utils import *
import glob

def load_model(VAE, model_path):
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


def run(VAE, model_path):
    if not os.path.exists('demo'):
        os.mkdir('demo')
    model_name = os.path.join('demo', model_path.split('\\')[-1][0: -3])
    if not os.path.exists(model_name):
        os.mkdir(model_name)
        
    model = load_model(VAE, model_path)
    m1 = "original-melody/jigs222.mid"
    c1 = "original-chord/" + m1.split('/')[1]
    m2 = "original-melody/reelsa-c22.mid"
    c2 = "original-chord/" + m2.split('/')[1]
    m3 = "original-melody/hpps55.mid"
    c3 = "original-chord/" + m1.split('/')[1]
    m4 = "original-melody/jigs33.mid"
    c4 = "original-chord/" + m2.split('/')[1]
    m5 = "original-melody/ashover13.mid"
    c5 = "original-chord/" + m2.split('/')[1]
    
    random.seed(22)
    sec1 = random.randint(1, 100)
    sec2 = random.randint(1, 100)
    sec3 = random.randint(1, 100)
    sec4 = random.randint(1, 100)
    sec4 = 25
    sec5 = random.randint(1, 100)
    start1 = int(sec1 * 8)
    start2 = int(sec2 * 8)
    start3 = int(sec3 * 8)
    start4 = int(sec4 * 8)
    start5 = int(sec5 * 8)
    length = 1
    
    melody1 = melody_to_numpy(fpath=m1)[start1: start1 + 32 * length]
    chord1 = chord_to_numpy(fpath=c1)[start1: start1 + 32 * length]
    melody2 = melody_to_numpy(fpath=m2)[start2: start2 + 32 * length]
    chord2 = chord_to_numpy(fpath=c2)[start2: start2 + 32 * length]
    melody3 = melody_to_numpy(fpath=m3)[start3: start3 + 32 * length]
    chord3 = chord_to_numpy(fpath=c3)[start3: start3 + 32 * length]
    melody4 = melody_to_numpy(fpath=m4)[start4: start4 + 32 * length]
    chord4 = chord_to_numpy(fpath=c4)[start4: start4 + 32 * length]
    melody5 = melody_to_numpy(fpath=m5)[start5: start5 + 32 * length]
    chord5 = chord_to_numpy(fpath=c5)[start5: start5 + 32 * length]
    
    chord1[chord1>1] = 1
    chord2[chord2>1] = 1
    chord3[chord3>1] = 1
    chord4[chord4>1] = 1
    chord5[chord5>1] = 1
    

#     # prepare 16th note C
#     melody16 = torch.zeros(32, 130)
#     melody16[:, 60] = 1.
    
#     # prepare empty chord
#     empty_chord = torch.zeros_like(chord1)
    
#     # prepare scale melody
#     scale = torch.zeros(32, 130)
#     scale[[0, 3, 4, 8, 11, 12, 16, 18, 20, 
#            21, 22, 23, 24, 25, 26, 27, 28], 
#           [67, 71, 74, 71, 74, 79, 78, 76, 
#            74, 76, 74, 72, 71, 72, 71, 69, 67]] = 1.
#     scale[[1, 2, 5, 6, 7, 9, 10, 13, 14, 15, 17, 19, 29, 30, 31], 128] = 1.
    
#     # prepare scale chord
#     scale_chord = torch.zeros(32, 12)
#     scale_chord[0: 16, [2, 7, 11]] = 1.
#     scale_chord[16: 24, [2, 6, 9]] = 1.
#     scale_chord[24: 32, [2, 7, 11]] = 1.
    
    # prepare shift one chord, 5 chords and 7 chords
    down_chord1_1 = torch.clone(chord1)
    down_chord1_1 = down_chord1_1[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]]
    down_chord1_5 = torch.clone(chord1)
    down_chord1_5 = down_chord1_5[:, [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]]
    down_chord1_7 = torch.clone(chord1)
    down_chord1_7 = down_chord1_7[:, [7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6]]
    
    down_chord2_1 = torch.clone(chord2)
    down_chord2_1 = down_chord2_1[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]]
    down_chord2_5 = torch.clone(chord2)
    down_chord2_5 = down_chord2_5[:, [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]]
    down_chord2_7 = torch.clone(chord2)
    down_chord2_7 = down_chord2_7[:, [7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6]]
    
    down_chord3_1 = torch.clone(chord3)
    down_chord3_1 = down_chord3_1[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]]
    down_chord3_5 = torch.clone(chord3)
    down_chord3_5 = down_chord3_5[:, [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]]
    down_chord3_7 = torch.clone(chord3)
    down_chord3_7 = down_chord3_7[:, [7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6]]
    
    down_chord4_1 = torch.clone(chord4)
    down_chord4_1 = down_chord4_1[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]]
    down_chord4_5 = torch.clone(chord4)
    down_chord4_5 = down_chord4_5[:, [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]]
    down_chord4_7 = torch.clone(chord4)
    down_chord4_7 = down_chord4_7[:, [7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6]]
    
    down_chord5_1 = torch.clone(chord5)
    down_chord5_1 = down_chord5_1[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]]
    down_chord5_5 = torch.clone(chord5)
    down_chord5_5 = down_chord5_5[:, [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]]
    down_chord5_7 = torch.clone(chord5)
    down_chord5_7 = down_chord5_7[:, [7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6]]
    
    # prepare minor chord
    minor_1 = torch.zeros(32, 12)
    minor_1[0:8, [2, 5, 9]] = 1.
    minor_1[8:16, :] = chord1[8:16, :]
    minor_1[16:24, [2, 5, 9]] = 1.
    minor_1[24:32, :] = chord1[24:32, :]
    
    minor_2 = torch.zeros(32, 12)
    minor_2[0:24, [2, 6, 11]] = 1.
    minor_2[24:32, :] = chord2[24:32, :]
    
    minor_3 = torch.zeros(32, 12)
    minor_3[0:24, [2, 5, 9]] = 1.
    minor_3[24:32, :] = chord3[24:32, :]
    
    minor_4 = torch.zeros(32, 12)
    minor_4[0:8, :] = chord4[0:8, :]
    minor_4[8:16, [2, 6, 11]] = 1.
    minor_4[16:24, :] = chord4[16:24, :]
    minor_4[24:32, [2, 6, 11]] = 1.
    
    minor_5 = torch.zeros(32, 12)
    minor_5[0:8, [2, 6, 11]] = 1.
    minor_5[8:16, :] = chord5[8:16, :]
    minor_5[16:24, [2, 6, 11]] = 1.
    minor_5[24:32, [0, 3, 7]] = 1.
    
    # save chords
    chord_to_midi(chord1, 'jigs222_chord.mid')
    chord_to_midi(down_chord1_1, 'mid/move_1/down_chord_1_jigs222.mid')
    chord_to_midi(down_chord1_5, 'mid/move_5/down_chord_5_jigs222.mid')
    chord_to_midi(down_chord1_7, 'mid/move_7/down_chord_7_jigs222.mid')
    
    chord_to_midi(chord2, 'reelsa-c22_chord.mid')
    chord_to_midi(down_chord2_1, 'mid/move_1/down_chord_1_reelsa-c22.mid')
    chord_to_midi(down_chord2_5, 'mid/move_5/down_chord_5_reelsa-c22.mid')
    chord_to_midi(down_chord2_7, 'mid/move_7/down_chord_7_reelsa-c22.mid')
    
    chord_to_midi(chord3, 'hpps55_chord.mid')
    chord_to_midi(down_chord3_1, 'mid/move_1/down_chord_1_hpps55.mid')
    chord_to_midi(down_chord3_5, 'mid/move_5/down_chord_5_hpps55.mid')
    chord_to_midi(down_chord3_7, 'mid/move_7/down_chord_7_hpps55.mid')
    
    chord_to_midi(chord4, 'jigs33_chord.mid')
    chord_to_midi(down_chord4_1, 'mid/move_1/down_chord_1_jigs33.mid')
    chord_to_midi(down_chord4_5, 'mid/move_5/down_chord_5_jigs33.mid')
    chord_to_midi(down_chord4_7, 'mid/move_7/down_chord_7_jigs33.mid')
    
    chord_to_midi(chord5, 'ashover13_chord.mid')
    chord_to_midi(down_chord5_1, 'mid/move_1/down_chord_1_ashover13.mid')
    chord_to_midi(down_chord5_5, 'mid/move_5/down_chord_5_ashover13.mid')
    chord_to_midi(down_chord5_7, 'mid/move_7/down_chord_7_ashover13.mid')
    
    chord_to_midi(minor_1, 'mid/major_to_minor/minor_jigs222.mid')
    chord_to_midi(minor_2, 'mid/major_to_minor/minor_reelsa-c22.mid')
    chord_to_midi(minor_3, 'mid/major_to_minor/minor_hpps55.mid')
    chord_to_midi(minor_4, 'mid/major_to_minor/minor_jigs33.mid')
    chord_to_midi(minor_5, 'mid/major_to_minor/minor_ashover13.mid')
    
    numpy_to_midi(melody1, output='mid/jigs222_melody.mid')
    numpy_to_midi(melody2, output='mid/reelsa-c22_melody.mid')
    numpy_to_midi(melody3, output='mid/hpps55_melody.mid')
    numpy_to_midi(melody4, output='mid/jigs33_melody.mid')
    numpy_to_midi(melody5, output='mid/ashover13_melody.mid')
    
    
#     rhy1 = swap(model, melody1, chord1, melody16, empty_chord)
#     rhy2 = swap(model, melody1, chord1, melody2, chord2)
#     pit1 = swap(model, scale, scale_chord, melody1, chord1)
#     pit2 = swap(model, melody2, chord2, melody1, chord1)
    down1_1 = swap(model, melody1, chord1, melody1, down_chord1_1, mode=2)
    down1_5 = swap(model, melody1, chord1, melody1, down_chord1_5, mode=2)
    down1_7 = swap(model, melody1, chord1, melody1, down_chord1_7, mode=2)
    down2_1 = swap(model, melody2, chord2, melody2, down_chord2_1, mode=2)
    down2_5 = swap(model, melody2, chord2, melody2, down_chord2_5, mode=2)
    down2_7 = swap(model, melody2, chord2, melody2, down_chord2_7, mode=2)
    down3_1 = swap(model, melody3, chord3, melody3, down_chord3_1, mode=2)
    down3_5 = swap(model, melody3, chord3, melody3, down_chord3_5, mode=2)
    down3_7 = swap(model, melody3, chord3, melody3, down_chord3_7, mode=2)
    down4_1 = swap(model, melody4, chord4, melody4, down_chord4_1, mode=2)
    down4_5 = swap(model, melody4, chord4, melody4, down_chord4_5, mode=2)
    down4_7 = swap(model, melody4, chord4, melody4, down_chord4_7, mode=2)
    down5_1 = swap(model, melody5, chord5, melody5, down_chord5_1, mode=2)
    down5_5 = swap(model, melody5, chord5, melody5, down_chord5_5, mode=2)
    down5_7 = swap(model, melody5, chord5, melody5, down_chord5_7, mode=2)
    minor1 = swap(model, melody1, chord1, melody1, minor_1, mode=2)
    minor2 = swap(model, melody2, chord2, melody2, minor_2, mode=2)
    minor3 = swap(model, melody3, chord3, melody3, minor_3, mode=2)
    minor4 = swap(model, melody4, chord4, melody4, minor_4, mode=2)
    minor5 = swap(model, melody5, chord5, melody5, minor_5, mode=2)
    
    items = [down1_1, down1_5, down1_7, down2_1, down2_5, down2_7, down3_1, down3_5, down3_7,
             down4_1, down4_5, down4_7, down5_1, down5_5, down5_7, minor1, minor2, minor3, minor4, minor5]
    temp_names = ['mid/move_1/' + model_path.rstrip('params.pt') + '_down_chord_j222_1',
                  'mid/move_5/' + model_path.rstrip('params.pt') + '_down_chord_j222_5',
                  'mid/move_7/' + model_path.rstrip('params.pt') + '_down_chord_j222_7',
                  'mid/move_1/' + model_path.rstrip('params.pt') + '_down_chord_reelsa-c22_1',
                  'mid/move_5/' + model_path.rstrip('params.pt') + '_down_chord_reelsa-c22_5',
                  'mid/move_7/' + model_path.rstrip('params.pt') + '_down_chord_reelsa-c22_7',
                  'mid/move_1/' + model_path.rstrip('params.pt') + '_down_chord_hpps55_1',
                  'mid/move_5/' + model_path.rstrip('params.pt') + '_down_chord_hpps55_5',
                  'mid/move_7/' + model_path.rstrip('params.pt') + '_down_chord_hpps55_7',
                  'mid/move_1/' + model_path.rstrip('params.pt') + '_down_chord_jigs33_1',
                  'mid/move_5/' + model_path.rstrip('params.pt') + '_down_chord_jigs33_5',
                  'mid/move_7/' + model_path.rstrip('params.pt') + '_down_chord_jigs33_7',
                  'mid/move_1/' + model_path.rstrip('params.pt') + '_down_chord_ashover13_1',
                  'mid/move_5/' + model_path.rstrip('params.pt') + '_down_chord_ashover13_5',
                  'mid/move_7/' + model_path.rstrip('params.pt') + '_down_chord_ashover13_7',
                  'mid/major_to_minor/' + model_path.rstrip('params.pt') + '_minor_chord_j222',
                  'mid/major_to_minor/' + model_path.rstrip('params.pt') + '_minor_chord_reelsa-c22',
                  'mid/major_to_minor/' + model_path.rstrip('params.pt') + '_minor_chord_hpps55',
                  'mid/major_to_minor/' + model_path.rstrip('params.pt') + '_minor_chord_jigs33',
                  'mid/major_to_minor/' + model_path.rstrip('params.pt') + '_minor_chord_ashover13']
    chords = [down_chord1_1, down_chord1_5, down_chord1_7, down_chord2_1, down_chord2_5, down_chord2_7,
              down_chord3_1, down_chord3_5, down_chord3_7, down_chord4_1, down_chord4_5, down_chord4_7,
              down_chord5_1, down_chord5_5, down_chord5_7, minor_1, minor_2, minor_3, minor_4, minor_5]
    
    for i, (it, tn, chord) in enumerate(zip(items, temp_names, chords)):
        if i <= 20:
            numpy_to_midi(torch.cat(tuple(it), 0), output=tn + '.mid')
        else:
            numpy_to_midi(it, output=tn + '.mid')
    

def batch_run():
#     models = glob.glob('models/*.pt')
#     for model_path in models:
    
#     from ec2vae import VAE
#     run(VAE, "params.pt")
#     from attn_ecvae_cq import VAE
#     run(VAE, "attn_cq_params.pt")
#     from attn_ecvae_mq import VAE
#     run(VAE, "attn_mq_params.pt")
    from embed_ecvae import VAE
    run(VAE, 'embed_params.pt')
    
    
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


###############################################################################

import random
from torch.distributions import Normal

def sample_m(model, model_path):
    model = load_model(model, model_path)
    m1 = "original-melody/jigs222.mid"
    c1 = "original-chord/" + m1.split('/')[1]
    m2 = "original-melody/reelsa-c22.mid"
    c2 = "original-chord/" + m2.split('/')[1]
    m3 = "original-melody/hpps55.mid"
    c3 = "original-chord/" + m1.split('/')[1]
    m4 = "original-melody/jigs33.mid"
    c4 = "original-chord/" + m2.split('/')[1]
    m5 = "original-melody/ashover13.mid"
    c5 = "original-chord/" + m2.split('/')[1]

    random.seed(22)
    sec1 = random.randint(1, 100)
    sec2 = random.randint(1, 100)
    sec3 = random.randint(1, 100)
    sec4 = random.randint(1, 100)
    sec5 = random.randint(1, 100)
    start1 = int(sec1 * 8)
    start2 = int(sec2 * 8)
    start3 = int(sec3 * 8)
    start4 = int(sec4 * 8)
    start5 = int(sec5 * 8)
    length = 1

    melody1 = melody_to_numpy(fpath=m1)[start1: start1 + 32 * length]
    chord1 = chord_to_numpy(fpath=c1)[start1: start1 + 32 * length]
    melody2 = melody_to_numpy(fpath=m2)[start2: start2 + 32 * length]
    chord2 = chord_to_numpy(fpath=c2)[start2: start2 + 32 * length]
    melody3 = melody_to_numpy(fpath=m3)[start3: start3 + 32 * length]
    chord3 = chord_to_numpy(fpath=c3)[start3: start3 + 32 * length]
    melody4 = melody_to_numpy(fpath=m4)[start4: start4 + 32 * length]
    chord4 = chord_to_numpy(fpath=c4)[start4: start4 + 32 * length]
    melody5 = melody_to_numpy(fpath=m5)[start5: start5 + 32 * length]
    chord5 = chord_to_numpy(fpath=c5)[start5: start5 + 32 * length]

    chord_to_midi(chord1.cpu(), 'j222_chord.mid')
    chord_to_midi(chord2.cpu(), 'r22_chord.mid')
    chord_to_midi(chord3.cpu(), 'hpps55_chord.mid')
    chord_to_midi(chord4.cpu(), 'jigs33_chord.mid')
    chord_to_midi(chord5.cpu(), 'ashover13_chord.mid')

    melody1 = torch.stack(melody1.split(32, 0)).float()
    melody2 = torch.stack(melody2.split(32, 0)).float()
    melody3 = torch.stack(melody3.split(32, 0)).float()
    melody4 = torch.stack(melody4.split(32, 0)).float()
    melody5 = torch.stack(melody5.split(32, 0)).float()
    chord1 = torch.stack(chord1.split(32, 0)).float()
    chord2 = torch.stack(chord2.split(32, 0)).float()
    chord3 = torch.stack(chord3.split(32, 0)).float()
    chord4 = torch.stack(chord4.split(32, 0)).float()
    chord5 = torch.stack(chord5.split(32, 0)).float()

    if torch.cuda.is_available():
        model = model.cuda()
        melody1 = melody1.cuda()
        melody2 = melody2.cuda()
        melody3 = melody3.cuda()
        melody4 = melody4.cuda()
        melody5 = melody5.cuda()
        chord1 = chord1.cuda()
        chord2 = chord2.cuda()
        chord3 = chord3.cuda()
        chord4 = chord4.cuda()
        chord5 = chord5.cuda()
    with torch.no_grad():
        o_p, o_r = model.encoder(melody1, chord1)
        z_mean = torch.zeros_like(o_p.mean)
        z_var = torch.ones_like(o_r.mean)
        zp = Normal(z_mean, z_var)
        zr = Normal(z_mean, z_var)
        z1p = zp.rsample()
        z2r = zr.rsample()
        output1 = model.decoder(z1p, z2r, chord1).cpu()
        z1p = zp.rsample()
        z2r = zr.rsample()
        output2 = model.decoder(z1p, z2r, chord2).cpu()
        z1p = zp.rsample()
        z2r = zr.rsample()
        output3 = model.decoder(z1p, z2r, chord3).cpu()
        z1p = zp.rsample()
        z2r = zr.rsample()
        output4 = model.decoder(z1p, z2r, chord4).cpu()
        z1p = zp.rsample()
        z2r = zr.rsample()
        output5 = model.decoder(z1p, z2r, chord5).cpu()

        numpy_to_midi(torch.cat(tuple(output1), 0), output='mid/sample/sample_' + model_path.rstrip('params.pt') + 'j222' + '.mid')
        numpy_to_midi(torch.cat(tuple(output2), 0), output='mid/sample/sample_' + model_path.rstrip('params.pt') + 'r22' + '.mid')
        numpy_to_midi(torch.cat(tuple(output3), 0), output='mid/sample/sample_' + model_path.rstrip('params.pt') + 'hpps55' + '.mid')
        numpy_to_midi(torch.cat(tuple(output4), 0), output='mid/sample/sample_' + model_path.rstrip('params.pt') + 'jigs33' + '.mid')
        numpy_to_midi(torch.cat(tuple(output5), 0), output='mid/sample/sample_' + model_path.rstrip('params.pt') + 'ashover13' + '.mid')

from ec2vae import VAE
sample_m(VAE, "params.pt")
from attn_ecvae_cq import VAE
sample_m(VAE, "attn_cq_params.pt")
from attn_ecvae_mq import VAE
sample_m(VAE, "attn_mq_params.pt")
