import pandas as pd

from src import db


class _Cursor:
    def __init__(self):
        self.copied_sql = None
        self.copied_csv = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def copy_expert(self, sql, buf):
        self.copied_sql = sql
        self.copied_csv = buf.read()


class _Connection:
    def __init__(self):
        self.cursor_obj = _Cursor()
        self.committed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True


def test_copy_df_writes_infinite_numbers_as_null_csv_fields():
    conn = _Connection()
    df = pd.DataFrame(
        {
            "period_end_date": ["2026-07-03", "2026-07-04"],
            "symbol": ["AAA", "BBB"],
            "revenue_yoy": [float("inf"), float("-inf")],
        }
    )

    written = db.copy_df(
        conn,
        df,
        "backtesting_minervini_screen_daily",
        ["period_end_date", "symbol", "revenue_yoy"],
    )

    assert written == 2
    assert conn.committed is True
    assert conn.cursor_obj.copied_csv == "2026-07-03,AAA,\n2026-07-04,BBB,\n"
