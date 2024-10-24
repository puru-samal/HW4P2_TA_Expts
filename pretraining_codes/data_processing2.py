import torch
import torchaudio.transforms as TaT
from hw_tokenizers2 import GTokenizer
import os
from tqdm import tqdm
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class SpeechDataset(torch.utils.data.Dataset):

  def __init__(self, root:str, partition:str, config:dict, tokenizer:GTokenizer, isTrainPartition:bool, subset:float=1.0):
    
    # general
    self.partition        = partition
    self.isTrainPartition = isTrainPartition
    self.config           = config
    self.tokenizer = tokenizer
    self.eos_token = tokenizer.EOS_TOKEN
    self.sos_token = tokenizer.SOS_TOKEN
    self.pad_token = tokenizer.PAD_TOKEN
    
    # paths | files
    self.fbank_dir   = os.path.join(root, self.partition, "fbank")
    self.text_dir    = os.path.join(root, self.partition, "text")
    self.fbank_files = sorted(os.listdir(self.fbank_dir))
    self.text_files  = sorted(os.listdir(self.text_dir))

    assert len(self.fbank_files) == len(self.text_files), "Number of fbank files and text files must be the same"
    subset           = int(subset * len(self.fbank_files))
    self.fbank_files = sorted(os.listdir(self.fbank_dir))[:subset]
    self.text_files  = sorted(os.listdir(self.text_dir))[:subset]


    self.length = len(self.fbank_files)
    self.fbanks, self.transcripts_shifted, self.transcripts_golden = [], [], []

    # load mfccs
    for file in tqdm(self.fbank_files, desc=f"Loading mfcc data for {partition}"):
      fbank = np.load(os.path.join(self.fbank_dir, file))
      self.fbanks.append(fbank)

    # load and encode transcripts
    for file in tqdm(self.text_files, desc=f"Loading transcript data for {partition}"):
      transcript = np.load(os.path.join(self.text_dir, file)).tolist()
      transcript = "".join(transcript)
      tokenized  = self.tokenizer.encode(transcript)
      self.transcripts_shifted.append(np.array([self.eos_token] + tokenized))
      self.transcripts_golden.append(np.array(tokenized + [self.eos_token]))

    assert len(self.fbanks) == len(self.transcripts_shifted) == len(self.transcripts_golden), "Number of fbanks, shifted transcripts, and golden transcripts must be the same"

    # precompute global stats for global mean and variance normalization
    if self.config["global_mvn"]:
      self.global_mean, self.global_std = self.compute_global_stats()

    # Torch Audio Transforms
    # time masking
    self.time_mask = TaT.TimeMasking(time_mask_param=config["specaug_conf"]["time_mask_width_range"], iid_masks=True)
    # frequency masking
    self.freq_mask = TaT.FrequencyMasking(freq_mask_param=config["specaug_conf"]["freq_mask_width_range"], iid_masks=True)


  def __len__(self):
    return self.length

  def __getitem__(self, idx):
      fbank = torch.FloatTensor(self.fbanks[idx])
      shifted_transcript = torch.LongTensor(self.transcripts_shifted[idx])
      golden_transcript = torch.LongTensor(self.transcripts_golden[idx])

      # Apply global mean and variance normalization if enabled
      if self.config["global_mvn"] and self.global_mean is not None and self.global_std is not None:
          fbank = (fbank - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)

      return fbank, shifted_transcript, golden_transcript

  def collate_fn(self, batch):
    # Prepare batch
    batch_fbank      = [i[0] for i in batch]
    batch_transcript = [i[1] for i in batch]
    batch_golden     = [i[2] for i in batch]

    lengths_fbank      = [len(i.T) for i in batch_fbank]  # Lengths of each F x T sequence
    lengths_transcript = [len(i) for i in batch_transcript]

    # transpose to T x F
    batch_fbank = [i.T for i in batch_fbank]

    # Pad sequences
    batch_fbank_pad      = pad_sequence(batch_fbank, batch_first=True, padding_value=self.pad_token)
    batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True, padding_value=self.pad_token)
    batch_golden_pad     = pad_sequence(batch_golden, batch_first=True, padding_value=self.pad_token)

    # transpose back to F x T
    batch_fbank_pad = batch_fbank_pad.transpose(1, 2)

    # do specaugment transforms
    if self.config["specaug"] and self.isTrainPartition:
      # shape should be B x 80 x T
      assert batch_fbank_pad.shape[1] == 80

      if self.config["specaug_conf"]["apply_freq_mask"]:
        batch_fbank_pad = self.freq_mask(batch_fbank_pad)

      if self.config["specaug_conf"]["apply_time_mask"]:
        batch_fbank_pad = self.time_mask(batch_fbank_pad)

    # transpose back to T x F
    batch_fbank_pad = batch_fbank_pad.transpose(1, 2)

    # Return the following values:
    # padded features, padded shifted labels, padded golden labels, actual length of features, actual length of the shifted labels
    return batch_fbank_pad, batch_transcript_pad, batch_golden_pad, torch.tensor(lengths_fbank), torch.tensor(lengths_transcript)

  def compute_global_stats(self):
      ''' Compute global mean and variance for all FBanks across the dataset '''
      print("Computing global mean and variance for normalization")

      all_fbanks = np.concatenate([fbank for fbank in self.fbanks], axis=1)  # Concatenate along time axis (F x T)
      global_mean = np.mean(all_fbanks, axis=1)  # Mean across time (per frequency bin)
      global_std = np.std(all_fbanks, axis=1) + 1e-20  # Standard deviation across time (per frequency bin)

      print(f"Computed global mean: {global_mean.shape}, global variance: {global_std.shape}")
      return torch.FloatTensor(global_mean), torch.FloatTensor(global_std)