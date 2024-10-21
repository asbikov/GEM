# GEM

A language model designed for System 2 (long) thinking.

System 2 thinking requires the model to operate for extended periods, generating longer sequences while making deliberate decisions about what information to retain and what to discard from memory. Crucially, when the model produces new results, it must be able to immediately "learn" from them. GEM builds on an undercelebrated achievement in AI: how the context of language models effectively solves the challenge of few-shot learning, at least partially. GEM aims to expand on this concept by implementing a very large "fast weights" memory module where new information is immediately learned and gradually forgotten over time. This approach could eventually be refined to include different types of memory - shorter-term, longer-term, and permanent - by varying how quickly different weights update during the writing process.

A challenge in implementing such a system lies in managing resource consumption during training. Traditional autograd approaches would require storing multiple intermediate states of the memory, making it impractical for long contexts. GEM solves this through a carefully designed memory module with read and write operations. The write operation is specifically engineered to be reversible, allowing the reconstruction of intermediate states during the backward pass without storing them explicitly. This reversibility, combined with the gradual forgetting property, enables efficient training while maintaining the model's ability to learn and forget information dynamically.

## 🚀 Installation

```bash
git clone https://github.com/asbikov/GEM.git
cd GEM
pip install -r requirements.txt
```

## 🚂 Model Training

Let's compare a small GEM model with a similarly sized transformer to make sure our implementation is actually working.

Check out Andrej Karpathy's awesome [nanoGPT](https://github.com/karpathy/nanoGPT)! We can run a similar test where a character-level GEM model with just under 1M parameters is trained on the works of Shakespeare. It achieves a validation loss of 1.8557, which is on par with the vanilla transformer.

```bash
python train.py
```

```
TrainConfig(device='cpu', dataset_name='karpathy/tiny_shakespeare', tokenizer_name='character', epochs=2, batch_size=1, gradient_accumulation_steps=12, learning_rate=0.001, checkpoints_dir='checkpoints', continue_from_checkpoint='')
GEMConfig(embedding_size=128, memory_size=64, vocabulary_size=None, sequence_length=64, n_layers=4)
Fitting tokenizer: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 62.52it/s]
Vocabulary size: 65
Total parameters: 882560
Epoch 1/2
Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 15685/15685 [3:38:46<00:00,  1.19it/s, moving_loss=1.9557] 
Validating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 871/871 [02:19<00:00,  6.26it/s, moving_loss=1.9794]
Train Loss: 2.2394, Val Loss: 2.0330
Epoch 2/2
Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 15685/15685 [3:21:38<00:00,  1.30it/s, moving_loss=1.7402]
Validating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 871/871 [02:07<00:00,  6.85it/s, moving_loss=1.7923]
Train Loss: 1.8289, Val Loss: 1.8557
```

## 📚 Text Generation

Generate text by sampling the trained model:

```bash
python sample.py
```

```
That of begricious with here well, by me straing to as borning soon.
To dead asting wead better be those with them,
And and beeave the the balinds begain sould beasoure the been farrem.

LEONTES:
We would muselfs soull ast and all them with borned.

First, and maind somer to thank my trother with a the some the fame since,
There bidstating susiness wead the weadon bard becoume must thou ting,
Then done
That stinerve with the calials, too did the gaies stried and stoon.

Fiven me swould is firends and are to steem.

LEONTES:
How madiom.

BENVOLIO:
Hast me marrcious on with bover thou backing somme all somme of deem
To mucke signingmenty attalt berove,
Than my despeak, the shalt is the came and with withn son.

PETILIDIUS:
Is beadous, lies, more offest of them subatie and the prestage,
To mostated the mown'd on for the be busing beread the the sir;
And is not them be weaken sthorer, stho sting and my becove.

And she have not were best the take botter, and stake them that herefore
Fore the badistalt my becomer,
```

Drunk Shakespeare. 

## 📋 ToDo

- [ ] Find why weight tying of the last layer is not working. It may reveal a bug.
- [ ] Some kind of paralelization is necessary.
- [ ] Scaling up - MoE approach.
- [ ] README - model architecture.
- [ ] README - math.
- [ ] README - references.
