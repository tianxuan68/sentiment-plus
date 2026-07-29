#!/usr/bin/env python3
"""Check sentiment menus (sa010-sa015) in sys_permission."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from app.db.session import SessionLocal


def main() -> int:
    expected = ["sa010", "sa011", "sa012", "sa013", "sa014", "sa015"]
    with SessionLocal() as db:
        rows = db.execute(
            text(
                "SELECT id, name, url, component FROM sys_permission "
                "WHERE id LIKE 'sa%' ORDER BY id"
            )
        ).fetchall()
        found = {r[0] for r in rows}
        print(f"Found {len(rows)} sentiment menu rows:")
        for r in rows:
            print(f"  {r[0]} | {r[1]} | {r[2]} | {r[3]}")

        missing = [x for x in expected if x not in found]
        if missing:
            print("\nMISSING:", ", ".join(missing))
            print("Run: mysql ... < sql/patch_add_sentiment_menus.sql")
            return 1

        user_count = db.execute(text("SELECT COUNT(*) FROM sys_user")).scalar()
        print(f"\nOK  all sa010-sa015 present; sys_user count = {user_count}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
