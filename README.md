# Darija Transcription

Application de transcription vocale en arabe marocain, basée sur un modèle Whisper Small fine-tuné.

## Structure du projet

```
your-repo/
  server.py
  index.html
  requirements.txt
  whisper-small-darija/       # à télécharger séparément
    config.json
    model.safetensors
    tokenizer.json
    ...
  Pipeline/
    audio_pipeline.py
    youtube_pipeline.py
    train.py
    pipeline_requirements.txt
```

## Lancer l'application

### Prérequis

Python 3.10 ou plus, le dossier `whisper-small-darija` téléchargé depuis Google Drive, et `ffmpeg` installé sur votre machine.

### 1. Cloner le repo

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Télécharger le modèle

Télécharger le dossier `whisper-small-darija` depuis [Google Drive](https://drive.google.com/file/d/1ASnNHSsGic3vFRB3eZ6tQQBVHJw-QkXY/view?usp=sharing) et le placer à la racine du projet comme indiqué dans la structure ci-dessus.

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer le serveur

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

L'application sera disponible sur `http://localhost:8000`.

### Support GPU

Si votre machine possède un GPU compatible CUDA, le modèle l'utilisera automatiquement sans configuration supplémentaire.

### API

| Méthode | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Vérifier l'état du serveur et du modèle |
| POST | `/transcribe` | Transcrire un fichier audio |

L'endpoint `/transcribe` accepte du `multipart/form-data` avec un champ `file`. Formats supportés : `wav`, `mp3`, `ogg`, `flac`, `m4a`, `webm`. Taille maximale : 50 Mo, durée maximale : 5 minutes.

## Fine-tuning du modèle

Le script `Pipeline/train.py` permet de fine-tuner `openai/whisper-small` sur des données en Darija.

### Sources de données supportées

**DODa (recommandé)** — Mettre `USE_DODA = True` et `USE_LOCAL = False` dans `train.py`. Télécharger les 5 fichiers parquet du [dataset DODa sur Kaggle](https://www.kaggle.com/datasets/youneseloiarm/moroccan-darija-voice-dataset) et les placer dans un dossier `doda/` à côté de `train.py`. Le script utilise la colonne `darija_Arab_new` comme cible de transcription.

**Clips locaux** — Mettre `USE_LOCAL = True` et `USE_DODA = False`. Pointer `CLIPS_DIR` vers votre dossier de clips contenant un fichier `metadata.jsonl` (produit par `youtube_pipeline.py`).

### Configuration d'entraînement

| Paramètre | Valeur |
|-----------|-------|
| Modèle de base | `openai/whisper-small` |
| Langue | Arabe |
| Tâche | Transcription |
| Batch size par appareil | 8 |
| Gradient accumulation | 2 (batch effectif : 16) |
| Learning rate | 1e-5 |
| Warmup steps | 500 |
| Max steps | 8000 |
| Métrique | WER (Word Error Rate) |
| Précision | FP16 |
| Sauvegarde | Tous les 1000 steps (garde les 3 derniers) |

L'entraînement reprend automatiquement depuis `./whisper-small-darija/checkpoint-2000` si ce dossier existe. Le meilleur checkpoint (WER le plus bas) est chargé à la fin et sauvegardé dans `./whisper-small-darija`.

Le dataset prétraité est mis en cache dans `./cached_dataset/` après la première exécution, ce qui évite de refaire le prétraitement lors des runs suivants.

### Installer les dépendances d'entraînement

```bash
cd Pipeline
pip install -r pipeline_requirements.txt
```

### Lancer l'entraînement

```bash
python train.py
```

### Notes

Sur Windows, mettre `dataloader_num_workers=0` car le multiprocessing est peu fiable. En cas d'erreur CUDA out of memory, décommenter `model.freeze_encoder()` dans `train.py` pour réduire l'utilisation VRAM. Les options `tf32=True` et `optim="adamw_torch_fused"` sont activées pour un entraînement plus rapide sur les GPU Ampere et plus récents.

## Pipeline de données

Le dossier `Pipeline/` contient les scripts pour construire un dataset d'entraînement à partir de vidéos YouTube.

### Comment ça marche

`youtube_pipeline.py` prend une liste d'URLs YouTube. Pour chaque vidéo, il télécharge l'audio et les sous-titres en arabe, découpe l'audio en courts clips alignés sur les timestamps des sous-titres, applique un pipeline de nettoyage sur chaque clip, et sauvegarde tout dans un fichier `metadata.jsonl` prêt pour l'entraînement.

`audio_pipeline.py` gère le nettoyage audio de chaque clip : conversion en WAV mono 16 kHz, réduction du bruit de fond, suppression des silences en début et fin de clip, et normalisation du volume.

### Prérequis

`ffmpeg` installé sur votre machine.

### Installer les dépendances du pipeline

```bash
cd Pipeline
pip install -r pipeline_requirements.txt
```

### Lancer le pipeline

Modifier la liste `urls` en bas de `youtube_pipeline.py`, puis lancer :

```bash
python youtube_pipeline.py
```

Les clips et le fichier manifest seront sauvegardés dans `./clips/` par défaut.

### Format de sortie

Le fichier `clips/metadata.jsonl` contient une entrée JSON par ligne :

```json
{"file_name": "VIDEO_ID/clip_00001.wav", "transcription": "النص هنا"}
```

Ce format est compatible avec les datasets Hugging Face pour le fine-tuning.