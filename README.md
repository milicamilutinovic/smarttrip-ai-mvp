# SmartTrip – Group Travel Recommender System

SmartTrip je AI MVP aplikacija za preporuku turističkih destinacija za grupe korisnika. 
Sistem koristi content-based pristup i grupnu agregaciju preferencija kako bi predložio optimalne destinacije na osnovu interesovanja, budžeta i drugih kriterijuma.

## Pokretanje aplikacije

### 1. Kloniranje repozitorijuma

git clone https://github.com/milicamilutinovic/smarttrip-ai-mvp.git
cd smarttrip-ai-mvp

### 2. Kreiranje i aktivacija virtualnog okruženja

Mac/Linux:

python -m venv .venv
source .venv/bin/activate

Windows:

python -m venv .venv
.venv\Scripts\activate

### 3. Instalacija zavisnosti

pip install -r requirements.txt

### 4. Pokretanje backend servera

uvicorn backend.api.main:app --reload

Aplikacija će biti dostupna na:

http://127.0.0.1:8000

API dokumentacija:

http://127.0.0.1:8000/docs
