# Traffy-Caeser

Final Project for DSDE

## Perequisite

- Python 3.11+
- UV

## Run local server

1. clone repository

```
git clone https://github.com/CP-RektMart/Traffy-Caeser

cd Traffy-Caeser
```

2. activate venv

```
uv venv

.venv\Scripts\activate
```

3. run docker compose

```
docker-compose up -d
```

4. start server

```
# Run main script
uv run main.py
```

```
# Run visualization
uv run dv
or dv
```

## Project Structure
```
.
├── README.md
├── data
├── gcp --> Data Engineering
├── main.py
├── notebooks --> Data Science
├── pyproject.toml
├── src --> Data Visualization
└── uv.lock
```