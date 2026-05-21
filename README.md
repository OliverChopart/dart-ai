# dart-ai

AI-drevet dartscoring til 501 — kameraet er en iPhone, detektionen kører lokalt på Apple Silicon.

## Hvordan det virker

Et YOLOv11-model kører live på kamerastrømmen og detekterer fem klasser: selve dartpilen (`dart`) og fire kalibreringspunkter på dobbeltringen (`cal_20`, `cal_6`, `cal_3`, `cal_11`). De fire kalibreringspunkter bruges til at beregne en homografi der folder skivens perspektiv ud til et kanonisk top-down-billede. Pile-koordinater mappes derefter til skivens geometri og scores i 501-spillet.

```
Kamerastrøm → YOLO-detektion → Homografi-beregning → Score-beregning → 501-spilmotor
```

## Hardwarekrav

- **Mac med Apple Silicon** (M1/M2/M3/M4) — modellen kører på MPS-backend
- **iPhone som webcam** via Apples [Continuity Camera](https://support.apple.com/en-us/102546)

> **Continuity Camera kræver:**
> - iPhone og Mac er logget ind på **samme Apple ID**
> - Bluetooth og Wi-Fi er slået til på begge enheder
> - iPhone er i nærheden af Mac'en
>
> Når disse betingelser er opfyldt, dukker iPhonen automatisk op som kamera-kilde i systemet (`CAMERA_SOURCE=0` eller `1` afhængig af om der er et andet kamera). Du behøver ikke installere noget.

Andre kamerakonfigurationer (EpocCam, USB-webcam) kan bruges, men er ikke testet grundigt.

## Installation

Projektet bruger [`uv`](https://docs.astral.sh/uv/) til pakkehåndtering.

```bash
# 1. Installer uv (hvis ikke allerede installeret)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Klon repo
git clone https://github.com/OliverChopart/dart-ai.git
cd dart-ai

# 3. Opret virtuelt miljø og installer dependencies
uv sync

# 4. Kopiér og tilpas konfiguration
cp .env.example .env
# Rediger .env efter behov (se Konfiguration nedenfor)
```

## Dependencies

Fra `pyproject.toml` (kræver Python ≥ 3.13):

| Pakke | Version |
|---|---|
| fastapi | ≥ 0.115.0 |
| uvicorn[standard] | ≥ 0.30.0 |
| websockets | ≥ 12.0 |
| pydantic | ≥ 2.7.0 |
| pydantic-settings | ≥ 2.3.0 |
| sqlalchemy[asyncio] | ≥ 2.0.30 |
| alembic | ≥ 1.13.0 |
| asyncpg | ≥ 0.29.0 |
| psycopg2-binary | ≥ 2.9.0 |
| opencv-python | ≥ 4.9.0 |
| numpy | ≥ 1.26.0 |
| torch | ≥ 2.3.0 |
| torchvision | ≥ 0.18.0 |
| ultralytics | ≥ 8.2.0 |
| python-dotenv | ≥ 1.0.0 |
| structlog | ≥ 24.0.0 |

Dev-dependencies: `pytest ≥ 8.0.0`, `pytest-asyncio ≥ 0.23.0`, `httpx ≥ 0.27.0`, `ruff ≥ 0.4.0`

## Konfiguration (.env)

Kopier `.env.example` til `.env` og tilpas:

```dotenv
# Kamera
CAMERA_SOURCE=0          # 0 = første kamera (typisk iPhone via Continuity Camera)
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=30

# Vision
YOLO_MODEL_PATH=models/yolo11n.pt   # skift til din trænede model efter træning
DETECTION_CONFIDENCE=0.5            # lavere = fanger flere, men flere fejl
DETECTION_DEVICE=mps                # mps = Apple Silicon GPU

# Database (bruges kun af backend-API, ikke af spillet)
DATABASE_URL=postgresql+asyncpg://localhost/dartai
```

Der er også avancerede indstillinger i `config/settings.py` der ikke er i `.env.example`:

| Variabel | Standard | Beskrivelse |
|---|---|---|
| `YOLO_CAL_CONFIDENCE` | `0.3` | Konfidenstærskel specifikt for kalibreringspunkter |
| `HOMOGRAPHY_FIFO_SIZE` | `5` | Antal frames der glides over ved homografi-stabilisering |
| `HOMOGRAPHY_FIFO_MIN_HITS` | `3` | Minimum frames med gyldig homografi inden den accepteres |
| `MAX_PLAYERS` | `4` | Maksimalt antal spillere |

## Opsætning af iPhone som Continuity Camera

1. Sørg for at iPhone og Mac er på **samme Apple ID** og har Bluetooth + Wi-Fi slået til
2. Placer iPhonen tæt på Mac'en — den dukker automatisk op som kamera-kilde
3. På nyere iPhones (12+) fungerer det uden ekstra app
4. Monter iPhonen stabilt over dartskiven med god vinkel — systemet kompenserer for perspektiv via homografien, men jo mere frontalt jo bedre
5. Verificer at iPhone dukker op: `uv run python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

Hvis du har både et internt kamera (MacBook) og en iPhone, prøv `CAMERA_SOURCE=1` eller `CAMERA_SOURCE=2`.

## Trin-for-trin: Træn din egen model

Du kan træne modellen udelukkende på dine egne billeder — anbefalet, da det giver det bedste resultat for dit specifikke kamera og opsætning.

### 1. Annoter billeder i Roboflow

1. Opret et projekt på [roboflow.com](https://roboflow.com) med **Object Detection**
2. Upload billeder af din dartskive med pile
3. Annoter med disse **5 klasser i præcis denne rækkefølge** (rækkefølgen er kritisk for model-output):

   | Klasse-ID | Navn | Hvad der annoteres |
   |---|---|---|
   | 0 | `dart` | Selve pilen (spids + skaft) |
   | 1 | `cal_20` | Dobbelt-20 feltet (toppen af skiven, ca. kl. 12) |
   | 2 | `cal_6` | Dobbelt-6 feltet (højre side, ca. kl. 4) |
   | 3 | `cal_3` | Dobbelt-3 feltet (venstre side, ca. kl. 8) |
   | 4 | `cal_11` | Dobbelt-11 feltet (højre-venstre, ca. kl. 10) |

4. Eksportér som **YOLOv8 format** — ZIP-filen indeholder `images/` og `labels/` mapper
5. Udpak til `dataset/own/`:
   ```
   dataset/own/
   ├── images/
   │   ├── IMG_001.jpg
   │   └── ...
   └── labels/
       ├── IMG_001.txt
       └── ...
   ```

> **Vigtigt:** Klasserne skal have præcis de navne og den rækkefølge som vist ovenfor. Eksportér fra Roboflow med `dart` som klasse 0 og `cal_11` som klasse 4.

> **Annoterings-note:** Der er en kendt uoverensstemmelse i annotationerne fra det originale McNally-datasæt — `cal_6` og `cal_11` er ombyttet i forhold til fysisk position. `vision/calibration.py` kompenserer for dette med justerede vinkler i homografi-beregningen. Hvis du annoterer dine egne billeder fra bunden, annotér efter det faktiske segment (ikke annotér efter McNally-konventionen).

### 2. Træn

```bash
# Download base-modellen (første gang)
curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -o models/yolo11n.pt

# Træn på dine egne billeder (anbefalet)
caffeinate -i uv run python scripts/train_5class.py --own-only

# Alternativt: træn med McNally-datasæt (kræver convert_dataset_5class.py kørt først)
caffeinate -i uv run python scripts/train_5class.py
```

`caffeinate -i` forhindrer Mac'en i at sove under træning.

Træning tager typisk 20-60 minutter afhængig af datasætstørrelse.

### 3. Opdatér .env

Når træning er færdig:

```dotenv
YOLO_MODEL_PATH=runs/train/dart_5class/weights/best.pt
```

### 4. Test detektionen

```bash
uv run python scripts/test_detector.py --random
```

## Start et spil

```bash
# Ét-spiller spil
uv run python scripts/play_501.py

# Flerspiller
uv run python scripts/play_501.py --players "Alice" "Bob"
uv run python scripts/play_501.py --players "Alice" "Bob" "Charlie" "Dave"

# Uden live preview (headless)
uv run python scripts/play_501.py --no-preview
```

### Tastatur-kommandoer

**I kamera-vinduet (OpenCV-vinduet):**

| Tast | Handling |
|---|---|
| `K` | Kalibrér (beregn homografi fra kalibreringspunkter) |
| `SPACE` | Score pil (registrér nuværende pil) |
| `ENTER` | Ny tur — fjern pile fra skiven |
| `Q` | Afslut |

**I terminalen:**

| Kommando | Handling |
|---|---|
| `ENTER` | Ny tur (fjern pile fra skiven) |
| `u` + `ENTER` | Fortryd sidste kast |
| `s` + `ENTER` | Vis scoreboard |
| `q` + `ENTER` | Afslut |

### Spil-workflow

1. Start `play_501.py` — kamera og model indlæses
2. Sørg for at hele dartskiven er synlig i kamera-vinduet
3. Tryk `K` (eller `SPACE`) for at kalibrere — vent på `✅ Kalibrering lykkedes!` i terminalen
4. Kast en pil → tryk `SPACE` → scoren vises
5. Kast endnu en pil → `SPACE` → osv. (op til 3 pile per tur)
6. Fjern pile fra skiven → tryk `ENTER` → næste spillers tur

## Manuel kalibrering (fallback)

Hvis modellen ikke kan detektere kalibreringspunkterne automatisk:

```bash
uv run python scripts/run_calibration.py
```

Det åbner et interaktivt vindue. Klik i rækkefølge: dobbelt-20 (top) → dobbelt-6 (højre) → dobbelt-3 (bund) → dobbelt-11 (venstre). Tryk `ENTER` for at gemme. Homografien gemmes i `config/homography.npy`.

## Mappestruktur

```
dart-ai/
├── backend/                  # Spilmotor og (fremtidig) API
│   ├── game/
│   │   ├── game_501.py       # 501-spilregler og -logik
│   │   ├── models.py         # Dataklasser (Player, Turn, osv.)
│   │   └── session.py        # Spilsession — binder kamera + spil sammen
│   └── api/                  # FastAPI-routes (ikke i aktiv brug)
├── config/
│   └── settings.py           # Pydantic-settings — alle konfigurationsvariabler
├── database/                 # SQLAlchemy-modeller og Alembic-migrationer
│   ├── connection.py
│   ├── models.py
│   └── migrations/
├── dataset/                  # Træningsdata (committes ikke — se .gitignore)
│   ├── own/                  # Dine egne annoterede billeder
│   │   ├── images/
│   │   └── labels/
│   └── yolo_5class/          # Konverteret McNally-datasæt (genereret)
├── models/                   # YOLO-modelfiler (.pt) — committes ikke
│   └── .gitkeep
├── scripts/                  # Kørselsscripts
│   ├── play_501.py           # ← Start her for at spille
│   ├── train_5class.py       # Træn 5-klasse-modellen
│   ├── convert_dataset_5class.py  # Konvertér McNally-datasæt
│   ├── run_calibration.py    # Manuel kalibrering
│   ├── run_stream.py         # Rå kamerastrøm til debugging
│   ├── test_detector.py      # Test YOLO-detektionen på billeder
│   └── test_game_501.py      # Unittest for spillogikken
├── utils/
│   ├── geometry.py           # Koordinat-transformationer
│   └── logging.py            # Struktureret logging via structlog
├── vision/
│   ├── calibration.py        # Homografi-beregning (auto + manuel)
│   ├── detector.py           # YOLO-wrapper
│   ├── pipeline.py           # Fuld vision-pipeline per frame
│   ├── scorer.py             # Koordinat → dartsegment-mapping
│   └── stream.py             # Kamerastrøm-håndtering
├── .env.example              # Konfigurationsskabelon
├── pyproject.toml            # Projektdefinition og dependencies
└── README.md
```

## Kendte begrænsninger

- **macOS-only:** OpenCV-vinduer skal køres på main-tråden på macOS — `play_501.py` er struktureret derefter og virker ikke uændret på Linux/Windows.
- **Homografi kræver 4 kalibreringspunkter:** Modellen skal detektere alle fire `cal_*`-klasser i samme frame for at beregne homografien. Hvis lyset er dårligt eller skiven er delvist skjult, falder kalibreringen tilbage til en gemt `config/homography.npy` (hvis den findes).
- **McNally-datasæt:** Der er en kendt annotation-uoverensstemmelse i McNally-datasættet — `cal_6` og `cal_11` er ombyttet i forhold til fysisk position. `vision/calibration.py` kompenserer med justerede vinkler. Egne annotationer bør følge det faktiske segment.
- **Database/API:** `backend/api/`, `database/` og `alembic.ini` er scaffolding til en fremtidig web-frontend — de bruges ikke af `play_501.py`.
- **Én model, ét kamera-setup:** Modellen er specifik for dit kamera-afstand og -vinkel. Ændrer du opstillingen markant, bør du genoptage og gettræne.

## Fejlfinding

**Kamera ikke fundet:**
```bash
# Tjek hvilke kamera-indekser der virker
uv run python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Kamera {i}: OK')
    cap.release()
"
```

**Kalibrering mislykkes (ikke nok kalibreringspunkter detekteret):**
- Sørg for at kalibreringspunkterne (dobbelt-20, -6, -3, -11) er synlige og ikke dækket af pile
- Prøv at sænke `YOLO_CAL_CONFIDENCE` til f.eks. `0.2` i `.env`
- Kør manuel kalibrering: `uv run python scripts/run_calibration.py`

**Model ikke fundet:**
```bash
# Download base-modellen
curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -o models/yolo11n.pt
```
Eller opdatér `YOLO_MODEL_PATH` i `.env` til din trænede model.
