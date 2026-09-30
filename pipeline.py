"""``python -m pipeline --input-dir PRO_SPs/ --output-dir dist/`` — see app/batch.py."""
import sys

from app.batch import main

if __name__ == "__main__":
    sys.exit(main())
