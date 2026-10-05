"""Each candidate is classified using exclusively its external context.

Offsets refer to Python Unicode code points, not bytes or UTF-16 code units.
The two recurrent encoders must remain independent and unidirectional.
"""
from dataclasses import asdict, dataclass
import copy
import math
from typing import List, Sequence, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

LABELS = ("NONE", "NAME", "PATH", "TEXT", "VALUE")
LABEL_IDS = {label: i for i, label in enumerate(LABELS)}


class ContextBanks(tuple):
    """Span banks with separate full-context banks for the count query."""
    def __new__(cls,left,right,global_banks):
        obj=super().__new__(cls,(left,right));obj.global_banks=global_banks;return obj


@dataclass
class ModelConfig:
    embedding_dim: int = 32
    hidden_dim: int = 96
    layers: int = 2
    classifier_dim: int = 64
    dropout: float = 0.1
    max_chars: int = 256
    auxiliary_heads: bool = False
    count_classes: int = 5
    count_feature_policy: str = 'global'


def encode_texts(texts: Sequence[str], device="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    width = max(1, int(lengths.max()))
    tokens = torch.zeros((len(texts), width, 4), dtype=torch.long)
    for row, text in enumerate(texts):
        encoded=[]
        for char in text:
            char_bytes=char.encode('utf-8')
            encoded.append([pos*257+byte+1 for pos,byte in enumerate(char_bytes)] + [0]*(4-len(char_bytes)))
        if encoded:
            tokens[row,:len(text)]=torch.tensor(encoded,dtype=torch.long)
    return tokens.to(device), lengths.to(device)


class ContextSpanModel(nn.Module):
    def __init__(self, config: ModelConfig = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.embedding = nn.Embedding(4 * 257, c.embedding_dim, padding_idx=0)
        kwargs = dict(input_size=c.embedding_dim, hidden_size=c.hidden_dim,
                      num_layers=c.layers, batch_first=True,
                      dropout=c.dropout if c.layers > 1 else 0)
        self.left_encoder = nn.GRU(**kwargs)
        self.right_encoder = nn.GRU(**kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(2 * c.hidden_dim, c.classifier_dim), nn.GELU(),
            nn.Dropout(c.dropout), nn.Linear(c.classifier_dim, len(LABELS)))
        self.boundary_mix=0.0
        self.local_window=0
        self.local_mix=0.0
        self.auxiliary=nn.ModuleDict()
        if c.count_classes not in (5,9): raise ValueError('Supported count vocabularies are 5 or 9 classes')
        if c.auxiliary_heads or c.count_classes==9:self.enable_auxiliary_heads()
        if c.count_classes==9:self.enable_exact_counts()

    def enable_auxiliary_heads(self):
        """Clone trained readout adapters, retaining the GRUs and span head."""
        if self.auxiliary:return
        self.auxiliary=nn.ModuleDict({key:copy.deepcopy(self.classifier) for key in ('count','start','end')})
        self.config.auxiliary_heads=True

    def query_head(self,key):
        return self.auxiliary[key] if key in self.auxiliary else self.classifier

    def enable_exact_counts(self):
        """Expand only the count readout to 0..8; retain candidate scores.

        Splitting the old 4-or-more mass equally preserves probabilities for
        0..3 at initialization. The new count classes are then post-trained.
        """
        self.enable_auxiliary_heads()
        old=self.auxiliary['count'][-1]
        if old.out_features==9:return
        head=nn.Linear(old.in_features,9).to(device=old.weight.device,dtype=old.weight.dtype)
        with torch.no_grad():
            head.weight[:4].copy_(old.weight[:4]);head.bias[:4].copy_(old.bias[:4])
            head.weight[4:].copy_(old.weight[4].expand(5,-1))
            head.bias[4:].copy_(old.bias[4]-math.log(5))
        self.auxiliary['count'][-1]=head
        self.config.count_classes=9

    def contexts(self, tokens: torch.Tensor, lengths: torch.Tensor):
        if bool((lengths < 1).any()):
            raise ValueError("Empty texts are handled by the harness, not the encoder")
        mask = (tokens != 0).unsqueeze(-1)
        embedded = (self.embedding(tokens) * mask).sum(dim=2)
        batch, width, _ = embedded.shape
        positions = torch.arange(width, device=tokens.device)[None, :]
        reverse_indices = (lengths[:, None] - 1 - positions).clamp(min=0)
        reversed_input = embedded.gather(1, reverse_indices[:, :, None].expand_as(embedded))
        valid = positions < lengths[:, None]
        reversed_input = reversed_input * valid[:, :, None]

        def run(encoder, inputs):
            packed = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True,
                                          enforce_sorted=False)
            out, _ = encoder(packed)
            return pad_packed_sequence(out, batch_first=True, total_length=width)[0]

        left_states = run(self.left_encoder, embedded)
        right_states = run(self.right_encoder, reversed_input)
        zero = embedded.new_zeros((batch, 1, self.config.hidden_dim))
        left_bank = torch.cat((zero, left_states), dim=1)
        reverse_bank = torch.cat((zero, right_states), dim=1)
        boundaries = torch.arange(width + 1, device=tokens.device)[None, :]
        suffix_indices = (lengths[:, None] - boundaries).clamp(min=0)
        right_bank = reverse_bank.gather(
            1, suffix_indices[:, :, None].expand(-1, -1, self.config.hidden_dim))
        global_banks=(left_bank,right_bank)
        if not self.local_window or not self.local_mix:return global_banks
        local_left,local_right=self.local_contexts(embedded,lengths)
        mix=self.local_mix
        return ContextBanks((1-mix)*left_bank+mix*local_left,(1-mix)*right_bank+mix*local_right,global_banks)

    def local_contexts(self,embedded,lengths):
        """Encode bounded exterior windows once per boundary, sharing GRUs.

        No candidate interior enters its start-prefix or end-suffix state.
        The sentence-count task retains the original full-context encoding.
        """
        batch,width,dim=embedded.shape;window=self.local_window
        boundary=torch.arange(width+1,device=embedded.device)[None,:].expand(batch,-1)
        offset=torch.arange(window,device=embedded.device)[None,None,:]
        left_length=boundary.clamp(max=window)
        right_length=(lengths[:,None]-boundary).clamp(min=0,max=window)
        def encode(encoder,sizes,indices):
            valid=(sizes>0)&(boundary<=lengths[:,None])
            clipped=indices.clamp(min=0,max=width-1)
            rows=torch.arange(batch,device=embedded.device)[:,None,None]
            inputs=embedded[rows,clipped]*(offset<sizes[:,:,None])[:,:,:,None]
            flat=inputs[valid];lens=sizes[valid]
            states=[]
            for begin in range(0,len(flat),2048):
                packed=pack_padded_sequence(flat[begin:begin+2048],lens[begin:begin+2048].cpu(),batch_first=True,enforce_sorted=False)
                _,h=encoder(packed);states.append(h[-1])
            result=embedded.new_zeros(batch,width+1,self.config.hidden_dim)
            if states:result[valid]=torch.cat(states)
            return result
        left_indices=(boundary-left_length)[:,:,None]+offset
        right_indices=(boundary+right_length-1)[:,:,None]-offset
        return encode(self.left_encoder,left_length,left_indices),encode(self.right_encoder,right_length,right_indices)

    def candidate_features(self, banks, candidates: torch.Tensor):
        """candidates is [N,3]: (batch index, start inclusive, end exclusive)."""
        left, right = banks
        row, start, end = candidates.unbind(dim=1)
        return torch.cat((left[row, start], right[row, end]), dim=-1)

    def score(self, banks, candidates: torch.Tensor, boundary=None):
        logits=self.classifier(self.candidate_features(banks, candidates))
        return self.blend_boundaries(logits,banks,candidates,boundary)

    def boundary_logits(self,banks):
        """Two learned prefix/suffix queries over exterior context only.

        A start sees only its prefix; an end sees only its suffix. Domain
        shifts distinguish these binary tasks from type/count classification.
        Older checkpoints reuse the span head; post-trained checkpoints may
        use cloned readout adapters. Neither query sees candidate interior.
        """
        left,right=banks;zero=torch.zeros_like(left)
        starts=self.query_head('start')(torch.cat((left,zero),-1)+4.)
        ends=self.query_head('end')(torch.cat((zero,right),-1)+6.)
        return starts,ends

    def blend_boundaries(self,logits,banks,candidates,boundary=None):
        if not self.boundary_mix:return logits
        starts,ends=boundary if boundary is not None else self.boundary_logits(banks);row,start,end=candidates.unbind(-1)
        evidence=(starts[row,start,1]-starts[row,start,0])+(ends[row,end,1]-ends[row,end,0])
        adjusted=logits.clone();adjusted[:,1:]+=self.boundary_mix*evidence[:,None]
        return adjusted

    def forward(self, tokens, lengths, candidates):
        return self.score(self.contexts(tokens, lengths), candidates)

    def count_features(self, banks, lengths, boundary=None):
        """Existing count readout input, optionally including neural edge evidence.

        Summaries use no words or syntax rules. Only this global count query
        receives the summaries; candidate classification remains exterior-only.
        """
        left, right = getattr(banks, 'global_banks', banks)
        row = torch.arange(len(lengths), device=lengths.device)
        features = torch.cat((left[row, lengths], right[row, 0]), -1) + 2.0
        if self.config.count_feature_policy == 'global':
            return features
        if self.config.count_feature_policy != 'boundaries' or features.shape[-1] < 32:
            raise ValueError('Unsupported count feature policy')
        starts, ends = boundary if boundary is not None else self.boundary_logits(banks)
        positions = torch.arange(starts.shape[1], device=lengths.device)[None, :]
        summaries = []
        for logits, mask in ((starts, positions < lengths[:, None]),
                             (ends, (positions > 0) & (positions <= lengths[:, None]))):
            probability = logits[..., :2].softmax(-1)[..., 1] * mask
            sums = [probability.sum(-1)/8, probability.square().sum(-1)/8,
                    probability.pow(4).sum(-1)/8]
            sums += [((probability > threshold) & mask).sum(-1)/8
                     for threshold in (.25, .5, .75, .9, .95)]
            top = probability.topk(min(8, probability.shape[1]), dim=-1).values
            top = torch.nn.functional.pad(top, (0, 8-top.shape[-1]))
            summaries.append(torch.cat((torch.stack(sums, -1), top), -1))
        return torch.cat((features[:, :-32], torch.cat(summaries, -1) + 2.), -1)

    def count_logits(self, banks, lengths, boundary=None):
        """A separate five-logit sentence-count query.

        Full sentence states are used only for global parameter count, never
        for candidate type scoring. The fixed shift separates this query
        domain from ordinary GRU features. Older checkpoints reuse the span
        head; auxiliary-head checkpoints have a separate readout adapter.
        Legacy classes mean 0,1,2,3,4-or-more; expanded readouts mean 0..8,
        irrespective of the separate candidate-type LABELS.
        """
        features = self.count_features(banks, lengths, boundary)
        return self.query_head('count')(features)

    @torch.inference_mode()
    def all_scores(self, texts: List[str], device="cpu", chunk_size=8192, with_counts=False, with_evidence=False):
        tokens, lengths = encode_texts(texts, device)
        banks = self.contexts(tokens, lengths)
        boundary=self.boundary_logits(banks) if self.boundary_mix or with_evidence else None
        counts = self.count_logits(banks, lengths, boundary).cpu() if with_counts else None
        results = []
        for row, text in enumerate(texts):
            pairs = [(s, e) for s in range(len(text)) for e in range(s + 1, len(text) + 1)]
            logits = []
            for begin in range(0, len(pairs), chunk_size):
                part = pairs[begin:begin + chunk_size]
                candidates = torch.tensor([(row, s, e) for s, e in part],
                                          dtype=torch.long, device=device)
                logits.append(self.score(banks, candidates,boundary).cpu())
            item = (pairs, torch.cat(logits))
            item=item + (counts[row],) if with_counts else item
            if with_evidence:
                item=item+({'start':boundary[0][row,:len(text)+1].cpu(),
                            'end':boundary[1][row,:len(text)+1].cpu()},)
            results.append(item)
        return results

    def metadata(self):
        return {"config": asdict(self.config), "labels": list(LABELS),
                "parameters": sum(p.numel() for p in self.parameters())}
