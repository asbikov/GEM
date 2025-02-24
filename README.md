# GEM

Exploring ideas for making an efficient language model with large memory.


## 📋 ToDo

- [*] [TTT layers](https://arxiv.org/pdf/2407.04620) for large trainable memory
- [ ] Mixture of Experts (MoE) for efficient scaling
- [ ] README - references.


## 🛠 Setup

```bash
git clone https://github.com/asbikov/GEM.git
cd GEM
pip install -r requirements.txt
```

## 🚂 Model Training

Let's compare a small GEM model with a similarly sized transformer to make sure our implementation is actually working.

Check out Andrej Karpathy's awesome [nanoGPT](https://github.com/karpathy/nanoGPT)! We can run a similar test where a character-level GEM model with just under 1M parameters is trained on the works of Shakespeare. It achieves a validation loss of 1.8057, which is on par with the vanilla transformer.

```bash
python train.py
```

```
TrainConfig(device='cuda', dataset_name='karpathy/tiny_shakespeare', tokenizer_name='./character_tokenizer', epochs=2, batch_size=1, gradient_accumulation_steps=12, learning_rate=0.001, checkpoints_dir='checkpoints', random_seed=1)
GEMConfig(vocabulary_size=None, sequence_length=64, minibatch_size=16, memory_size=64, embedding_size=128, n_layers=4)
Fitting tokenizer: 100%|█████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 58.73it/s]
Vocabulary size:  65
Total parameters: 874240
Epoch 1/2
Training: 100%|██████████████████████████████████████████████████████████| 15443/15443 [21:45<00:00, 11.83it/s, moving_loss=2.0223]
Validating: 100%|████████████████████████████████████████████████████████████| 857/857 [00:16<00:00, 50.78it/s, moving_loss=2.0722]
Train Loss: 2.2843, Val Loss: 2.1059
Epoch 2/2
Training: 100%|██████████████████████████████████████████████████████████| 15443/15443 [21:38<00:00, 11.89it/s, moving_loss=1.7106]
Validating: 100%|████████████████████████████████████████████████████████████| 857/857 [00:16<00:00, 50.50it/s, moving_loss=1.7590]
Train Loss: 1.8338, Val Loss: 1.8057
```

## 🚀 Generating Text

Generate text by sampling from the trained model::

```bash
python sample.py
```

```
SampleConfig(prompt='\n', device='cuda', checkpoint_path='checkpoints/best.pth', tokenizer_name='./character_tokenizer', max_tokens=1024, temperature=0.9, top_k=10, top_p=0.9, random_seed=1)
Fitting tokenizer: 100%|█████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 55.41it/s]


KING RICHARD II:
Marrchery bothings with hath do onclied have serving and thou will
Than beservion in old be of therefore, sting me which tongue and out of my mighty
Marcius one inceas and that almst we father,
To my life, but there armself, be be his beaster to do more
That the spirt merdent: we and make out of your for to comprait way
So with is thee thance to dosess we with thank the swarding,
And be a liftter that of welcond be that speak,
Wheret wan the marder to seal.

MENENE:
If, that the thath sead it my go say
Time, swears with will to the devil mannishere:
I marke mistren so thence alme a bruing seak:
Some, then my shold, my the revices here fairend sto the shalless:
Wear hastity the bessicelf with herdrait meeth: bray.

LADY CAPULET:
And that my lord many shall mettenly and with true;
With art in that my my gay
To that shomand a comprine thin to beseech'd anger him;
There with ather's more to my honoura my harged to make barried:
And besirely and our and be me his honourable.
That my that a the re
```

Drunk Shakespeare. 

