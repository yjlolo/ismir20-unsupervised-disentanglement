from pathlib import Path
import numpy as np
import librosa
from torch.utils.data import Dataset


class SOL_ordinario(Dataset):
    """The dataset class for SOL-ordinario (sol/acidsInstruments-ordinario/data)

    This dataset class can also handle path_to_data for ['.npy', '.pth'],
    change target_ext=['.npy'] or ['.pth'] to load waveforms stored as Numpy array or Torch tensor.
    """
    def __init__(self, path_to_data, target_instrument=['Brass', 'Keyboards_fix', 'Strings', 'Winds'],
                 target_ext=['.wav', '.aif'], transform=None):

        path_to_data = Path(path_to_data)
        if not path_to_data.exists():
            raise FileNotFoundError
        
        audio_files = []
        for f in path_to_data.glob("*"):
            if target_ext == ['.wav', '.aif']:
                if f.stem in target_instrument:
                    audio_files.extend([i for i in f.rglob("*") if i.suffix in target_ext])
            else:
                if f.suffix in target_ext:
                    audio_files.append(f)
        audio_files = sorted(audio_files)
        
        attributes = ['instrument', 'family', 'pitch', 'pitch_class', 'dynamic']
        dict_label = {k: [] for k in attributes}
        for i in audio_files:
            instrument, _, pitch, dynamic = i.stem.split('-')[:4]
            dict_label['instrument'].append(instrument_map[instrument])
            dict_label['family'].append(family_map[instrument])
            dict_label['pitch_class'].append(pitchclass_map[extract_pitchclass(pitch)])
            dict_label['dynamic'].append(dynamic_map[dynamic])
            dict_label['pitch'].append(pitch)

        pitches = list(set(dict_label['pitch']))
        sort_freq = np.argsort(np.array([librosa.note_to_hz(i) for i in pitches]))
        pitch_map = {pitches[i]: n for n, i in enumerate(sort_freq)}
        dict_label['pitch'] = [pitch_map[p] for p in dict_label['pitch']]
        # pitch_map = {v: k for k, v in enumerate(sorted(set(dict_label['pitch'])))}
        # dict_label['pitch'] = [pitch_map[p] for p in dict_label['pitch']]

        self.transform = transform
        self.path_to_audio = audio_files
        self.dict_label = dict_label
        self.instrument_map = instrument_map
        self.family_map = family_map
        self.pitch_map = pitch_map
        self.pitchclass_map = pitchclass_map
        self.dynamic_map = dynamic_map

    def __len__(self):
        return len(self.path_to_audio)

    def __getitem__(self, idx):
        x = str(self.path_to_audio[idx])
        y_ins = self.dict_label['instrument'][idx]
        y_p = self.dict_label['pitch'][idx]
        y_pc = self.dict_label['pitch_class'][idx]
        y_dyn = self.dict_label['dynamic'][idx]
        y_fam = self.dict_label['family'][idx]
   
        if self.transform:
            x = self.transform(x)

        return x, idx, [y_ins, y_p, y_pc, y_dyn, y_fam]

instrument_map = {
    # English-Horn
    'EH_nA': 0,
    # French-Horn
    'Hn': 1,
    'Corf': 1,
    'Corm': 1,
    # Tenor-Trombone
    'trbt': 2,
    # Trumpet-C
    'TpC': 3,
    'trof': 3,
    'trom': 3,
    'trop': 3,
    # Piano
    'Pno': 4,
    # Violin
    'Vn': 5,
    # Cello
    'Vc': 6,
    # Alto-Sax
    'ASax': 7,
    # Bassoon
    'Bn': 8,
    'fagf': 8,
    'fagm': 8,
    'fagp': 8,
    # Clarinet
    'ClBb': 9,
    'clbb': 9,
    # Flute
    'Fl': 10,
    # Oboe
    'Ob': 11
}

family_map = {
    'Fl': 0,
    'ASax': 0,
    'Ob': 0,
    'Bn': 0,
    'fagf': 0,
    'fagm': 0,
    'fagp': 0,
    'ClBb': 0,
    'clbb': 0,
    'EH_nA': 1,
    'TpC': 1,
    'trof': 1,
    'trom': 1,
    'trop': 1,
    'Hn': 1,
    'Corf': 1,
    'Corm': 1,
    'trbt': 1,
    'Vn': 2,
    'Vc': 2,
    'Pno': 3
}
     

def extract_pitchclass(p):
    if len(p) == 2:
        return p[0]
    else:
        return p[0:2]

pitchclass_map = {
    'A': 0,
    'A#': 1,
    'B': 2,
    'C': 3,
    'C#': 4,
    'D': 5,
    'D#': 6,
    'E': 7,
    'F': 8,
    'F#': 9,
    'G': 10,
    'G#': 11
}

dynamic_map = {
    'f': 0,
    'ff': 1,
    'mf': 2,
    'p': 3,
    'pp': 4
}
