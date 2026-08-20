import db
import data_fetch


def main():
    db.init_db()
    records = data_fetch.fetch_latest_gkg_tone()
    inserted = db.insert_news_records(records)
    print(f"[poll_news] fetched {len(records)} records, {inserted} new")


if __name__ == "__main__":
    main()
