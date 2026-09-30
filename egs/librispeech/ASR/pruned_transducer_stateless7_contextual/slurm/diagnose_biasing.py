#!/usr/bin/env python3
"""
Check whether the trained biasing modules do anything.

For a few test-clean batches with the predefined biasing lists, prints
  - the RMS of the encoder/decoder biasing outputs relative to the
    encoder/decoder outputs they are added to,
  - how much attention goes to the <no-bias> slot vs. the real words,
  - the transducer loss with and without biasing (biasing should lower it),
and, per checkpoint, how far the biasing parameters moved from epoch 1.

Run from egs/librispeech/ASR, e.g.
  python pruned_transducer_stateless7_contextual/slurm/diagnose_biasing.py \
    --exp-dir pruned_transducer_stateless7_contextual/exp --epochs 1,5,10
"""
import sys
from pathlib import Path

import k2
import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr_datamodule import LibriSpeechAsrDataModule  # noqa: E402
from context_collector import ContextCollector  # noqa: E402
from train import get_params, get_parser, get_transducer_model  # noqa: E402

from icefall.utils import add_sos  # noqa: E402

BIASING = ("context_encoder.", "encoder_biasing_adapter.", "decoder_biasing_adapter.")


def main():
    parser = get_parser()  # --exp-dir, --bpe-model, --context-dir, --n-distractors, model args
    parser.add_argument("--epochs", type=str, default="1,10")
    parser.add_argument("--num-batches", type=int, default=5)
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.return_cuts = True
    args.exp_dir = Path(args.exp_dir)
    args.context_dir = Path(args.context_dir)
    args.max_duration = 300

    params = get_params()
    params.update(vars(args))
    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    collector = ContextCollector(
        path_is21_deep_bias=params.context_dir, sp=sp,
        is_predefined=True, n_distractors=params.n_distractors,
    )
    dl = LibriSpeechAsrDataModule(args).test_dataloaders(
        LibriSpeechAsrDataModule(args).test_clean_cuts()
    )
    batches = []
    for i, b in enumerate(dl):
        if i >= params.num_batches:
            break
        batches.append(b)

    model = get_transducer_model(params).to(device)
    model.params = params
    model.eval()

    first = None
    for epoch in [int(e) for e in params.epochs.split(",")]:
        ckpt = torch.load(params.exp_dir / f"epoch-{epoch}.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        sd = {k: v.float() for k, v in model.state_dict().items() if k.startswith(BIASING)}
        if first is None:
            first = {k: v.clone() for k, v in sd.items()}
        moved = torch.stack([(sd[k] - first[k]).norm() / (first[k].norm() + 1e-8) for k in sd]).mean()
        print(f"\n=== epoch {epoch}: mean relative change of biasing params vs first epoch = {moved:.4f}")

        stats = {"enc_ratio": [], "dec_ratio": [], "nobias_attn": [], "loss_b": [], "loss_nb": [],
                 "true_attn": [], "true_uniform": []}
        with torch.no_grad():
            for b in batches:
                x = b["inputs"].to(device)
                x_lens = b["supervisions"]["num_frames"].to(device)
                word_list, word_lengths, n_words = collector.get_context_word_list(b)
                contexts = {"mode": "get_context_word_list", "word_list": word_list.to(device),
                            "word_lengths": word_lengths, "num_words_per_utt": n_words}
                y = k2.RaggedTensor(sp.encode(b["supervisions"]["text"], out_type=int)).to(device)

                enc, enc_lens = model.encoder(x, x_lens)
                h, mask = model.context_encoder.embed_contexts(contexts)
                enc_b, attn = model.encoder_biasing_adapter(enc, h, mask, need_weights=True)
                stats["enc_ratio"].append((enc_b.pow(2).mean().sqrt() / enc.pow(2).mean().sqrt()).item())
                stats["nobias_attn"].append(attn[..., 0].mean().item())

                # Attention mass on the words of the list that occur in the utterance
                # (same sorted order as get_context_word_list), vs. uniform attention.
                lists = [sorted(w) for w in collector._get_predefined_word_lists(b)]
                for i, (words, text) in enumerate(zip(lists, b["supervisions"]["text"])):
                    idx = [j + 1 for j, w in enumerate(words) if w in set(text.split())]
                    if idx:
                        T = enc_lens[i].item()
                        stats["true_attn"].append(attn[i, :T][:, idx].sum(-1).max().item())
                        stats["true_uniform"].append(len(idx) / (len(words) + 1))

                sos_y = add_sos(y, sos_id=params.blank_id).pad(mode="constant", padding_value=params.blank_id)
                dec = model.decoder(sos_y)
                dec_b, _ = model.decoder_biasing_adapter(dec, h, mask)
                stats["dec_ratio"].append((dec_b.pow(2).mean().sqrt() / dec.pow(2).mean().sqrt()).item())

                frames = (enc_lens.sum()).item()
                simple_b, pruned_b = model(x=x, x_lens=x_lens, y=y, contexts=contexts, prune_range=5)
                enc_ad, dec_ad = model.encoder_biasing_adapter.forward, model.decoder_biasing_adapter.forward
                model.encoder_biasing_adapter.forward = lambda q, *a, **k: (torch.zeros_like(q), None)
                model.decoder_biasing_adapter.forward = lambda q, *a, **k: (torch.zeros_like(q), None)
                simple_nb, pruned_nb = model(x=x, x_lens=x_lens, y=y, contexts=contexts, prune_range=5)
                model.encoder_biasing_adapter.forward, model.decoder_biasing_adapter.forward = enc_ad, dec_ad
                stats["loss_b"].append((0.5 * simple_b + pruned_b).item() / frames)
                stats["loss_nb"].append((0.5 * simple_nb + pruned_nb).item() / frames)

        mean = {k: sum(v) / len(v) for k, v in stats.items()}
        print(f"RMS(encoder biasing out) / RMS(encoder out) = {mean['enc_ratio']:.4f}")
        print(f"RMS(decoder biasing out) / RMS(decoder out) = {mean['dec_ratio']:.4f}")
        print(f"attention on <no-bias> slot (encoder side)  = {mean['nobias_attn']:.3f}")
        print(f"max over frames of attention on the utterance's biased words = {mean['true_attn']:.3f} "
              f"(uniform attention would give {mean['true_uniform']:.3f})")
        print(f"loss per frame: with biasing {mean['loss_b']:.4f}, without {mean['loss_nb']:.4f}")


if __name__ == "__main__":
    main()
