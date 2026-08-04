#!/bin/zsh
set -e

cd "${0:A:h}"

if [[ ! -x ".venv/bin/python" ]]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
fi

.venv/bin/python main.py
exec .venv/bin/python -m streamlit run dashboard/Home.py
